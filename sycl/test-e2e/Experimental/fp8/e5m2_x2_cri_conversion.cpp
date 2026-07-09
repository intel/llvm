// REQUIRES: intel_feature_gpu_cri
// RUN: %{build} -Xclang -freg-struct-return -Xspirv-translator=spir64 --spirv-ext=+SPV_INTEL_fp_conversions,+SPV_EXT_float8,+SPV_KHR_bfloat16 -o %t.out
// RUN: %{run} SYCL_UR_TRACE=1 %t.out

// UNSUPPORTED: target-nvidia, target-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with CRI driver, and the
// SPIR-V backend does not support the required SPIR-V extensions

// XFAIL: new-offload-model
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/22372

#include <iostream>

#include <cmath>
#include <cstdint>
#include <limits>
#include <sycl/ext/oneapi/experimental/builtins.hpp>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>
#include <sycl/queue.hpp>
#include <sycl/usm.hpp>

using namespace sycl::ext::oneapi::experimental;

constexpr float E5M2MaxNormal = 57344.0f;

bool equal_or_both_nan(float actual, float expected) {
  if (std::isnan(expected))
    return std::isnan(actual);
  return actual == expected;
}

bool equal_with_zero_sign(float actual, float expected) {
  if (!equal_or_both_nan(actual, expected))
    return false;
  if (expected == 0.0f)
    return std::signbit(actual) == std::signbit(expected);
  return true;
}

bool is_positive_infinity(float value) {
  return std::isinf(value) && !std::signbit(value);
}

bool is_negative_infinity(float value) {
  return std::isinf(value) && std::signbit(value);
}

template <typename T, bool UseMarray, saturation Sat>
int test_stochastic_constructor(sycl::queue &queue) {
  auto *out = sycl::malloc_shared<float>(2, queue);
  auto *seed = sycl::malloc_shared<uint32_t>(1, queue);
  seed[0] = 0x89abcdefu;

  queue.single_task([=]() {
    const float positive_input = std::numeric_limits<float>::infinity();
    const float negative_input = -std::numeric_limits<float>::infinity();
    const uint32_t initial_seed = seed[0];

    if constexpr (UseMarray) {
      sycl::marray<T, 2> input(static_cast<T>(positive_input),
                               static_cast<T>(negative_input));
      if constexpr (Sat == saturation::finite) {
        fp8_e5m2_x2 value(input, stochastic_seed(seed));
        sycl::marray<float, 2> unpacked =
            static_cast<sycl::marray<float, 2>>(value);
        out[0] = unpacked[0];
        out[1] = unpacked[1];
      } else {
        fp8_e5m2_x2 value(input, stochastic_seed(seed), saturation::none);
        sycl::marray<float, 2> unpacked =
            static_cast<sycl::marray<float, 2>>(value);
        out[0] = unpacked[0];
        out[1] = unpacked[1];
      }
    } else {
      T input[2] = {static_cast<T>(positive_input),
                    static_cast<T>(negative_input)};
      if constexpr (Sat == saturation::finite) {
        fp8_e5m2_x2 value(input, stochastic_seed(seed));
        sycl::marray<float, 2> unpacked =
            static_cast<sycl::marray<float, 2>>(value);
        out[0] = unpacked[0];
        out[1] = unpacked[1];
      } else {
        fp8_e5m2_x2 value(input, stochastic_seed(seed), saturation::none);
        sycl::marray<float, 2> unpacked =
            static_cast<sycl::marray<float, 2>>(value);
        out[0] = unpacked[0];
        out[1] = unpacked[1];
      }
    }
  });
  queue.wait_and_throw();

  int ret = 0;
  if constexpr (Sat == saturation::finite) {
    if (out[0] != E5M2MaxNormal)
      ret = 1;
    if (out[1] != -E5M2MaxNormal)
      ret = 1;
  } else {
    if (!is_positive_infinity(out[0]))
      ret = 1;
    if (!is_negative_infinity(out[1]))
      ret = 1;
  }

  sycl::free(out, queue);
  sycl::free(seed, queue);
  return ret;
}

template <typename T>
int test_explicit_to_even_carray_constructor(sycl::queue &queue) {
  T input[2] = {static_cast<T>(3.0517578125e-05f), static_cast<T>(-6.0f)};
  auto *data = sycl::malloc_shared<fp8_e5m2_x2>(1, queue);
  auto *out = sycl::malloc_shared<T>(2, queue);
  data[0] = fp8_e5m2_x2(input, rounding::to_even);

  queue.single_task([=]() {
    sycl::marray<T, 2> unpacked = static_cast<sycl::marray<T, 2>>(data[0]);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (static_cast<float>(out[0]) != 3.0517578125e-05f)
    ret = 1;
  if (static_cast<float>(out[1]) != -6.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

template <typename T>
int test_explicit_to_even_marray_constructor(sycl::queue &queue) {
  sycl::marray<T, 2> input(static_cast<T>(3.0f),
                           static_cast<T>(-1.52587890625e-05f));
  auto *data = sycl::malloc_shared<fp8_e5m2_x2>(1, queue);
  auto *out = sycl::malloc_shared<T>(2, queue);
  data[0] = fp8_e5m2_x2(input, rounding::to_even);

  queue.single_task([=]() {
    sycl::marray<T, 2> unpacked = static_cast<sycl::marray<T, 2>>(data[0]);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (static_cast<float>(out[0]) != 3.0f)
    ret = 1;
  if (static_cast<float>(out[1]) != -1.52587890625e-05f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_round_trip_nan(sycl::queue &queue) {
  const float input[2] = {std::numeric_limits<float>::quiet_NaN(),
                          -std::numeric_limits<float>::quiet_NaN()};
  auto *data = sycl::malloc_shared<fp8_e5m2_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e5m2_x2(input, rounding::to_even);

  queue.single_task([=]() {
    fp8_e5m2_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp8_e5m2_x2(unpacked, rounding::to_even);
    sycl::marray<float, 2> round_tripped =
        static_cast<sycl::marray<float, 2>>(data[0]);
    out[0] = round_tripped[0];
    out[1] = round_tripped[1];
  });
  queue.wait_and_throw();

  int ret = !(std::isnan(out[0]) && std::isnan(out[1]));
  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_round_trip_negative_zero(sycl::queue &queue) {
  const float input[2] = {-0.0f, 7.0f};
  auto *data = sycl::malloc_shared<fp8_e5m2_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e5m2_x2(input, rounding::to_even);

  queue.single_task([=]() {
    fp8_e5m2_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp8_e5m2_x2(unpacked, rounding::to_even);
    sycl::marray<float, 2> round_tripped =
        static_cast<sycl::marray<float, 2>>(data[0]);
    out[0] = round_tripped[0];
    out[1] = round_tripped[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (!equal_with_zero_sign(out[0], -0.0f))
    ret = 1;
  if (out[1] != 7.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_round_trip_subnormals(sycl::queue &queue) {
  const float input[2] = {3.0517578125e-05f, -4.57763671875e-05f};
  auto *data = sycl::malloc_shared<fp8_e5m2_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e5m2_x2(input, rounding::to_even);

  queue.single_task([=]() {
    fp8_e5m2_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp8_e5m2_x2(unpacked, rounding::to_even);
    sycl::marray<float, 2> round_tripped =
        static_cast<sycl::marray<float, 2>>(data[0]);
    out[0] = round_tripped[0];
    out[1] = round_tripped[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != 3.0517578125e-05f)
    ret = 1;
  if (out[1] != -4.57763671875e-05f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_round_trip_exact_normals(sycl::queue &queue) {
  const float input[2] = {57344.0f, 6.103515625e-05f};
  auto *data = sycl::malloc_shared<fp8_e5m2_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e5m2_x2(input, rounding::to_even);

  queue.single_task([=]() {
    fp8_e5m2_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp8_e5m2_x2(unpacked, rounding::to_even);
    sycl::marray<float, 2> round_tripped =
        static_cast<sycl::marray<float, 2>>(data[0]);
    out[0] = round_tripped[0];
    out[1] = round_tripped[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != 57344.0f)
    ret = 1;
  if (out[1] != 6.103515625e-05f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_round_trip_exact_subnormal_limits(sycl::queue &queue) {
  const float input[2] = {4.57763671875e-05f, 1.52587890625e-05f};
  auto *data = sycl::malloc_shared<fp8_e5m2_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e5m2_x2(input, rounding::to_even);

  queue.single_task([=]() {
    fp8_e5m2_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp8_e5m2_x2(unpacked, rounding::to_even);
    sycl::marray<float, 2> round_tripped =
        static_cast<sycl::marray<float, 2>>(data[0]);
    out[0] = round_tripped[0];
    out[1] = round_tripped[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != 4.57763671875e-05f)
    ret = 1;
  if (out[1] != 1.52587890625e-05f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_round_trip_saturation_and_infinity_clamp(sycl::queue &queue) {
  const float input[2] = {60000.0f, -std::numeric_limits<float>::infinity()};
  auto *data = sycl::malloc_shared<fp8_e5m2_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e5m2_x2(input, rounding::to_even);

  queue.single_task([=]() {
    fp8_e5m2_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp8_e5m2_x2(unpacked, rounding::to_even);
    sycl::marray<float, 2> round_tripped =
        static_cast<sycl::marray<float, 2>>(data[0]);
    out[0] = round_tripped[0];
    out[1] = round_tripped[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != 57344.0f)
    ret = 1;
  if (out[1] != -57344.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_infinity_no_saturation(sycl::queue &queue) {
  const float input[2] = {std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity()};
  auto *data = sycl::malloc_shared<fp8_e5m2_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e5m2_x2(input, rounding::to_even, saturation::none);

  queue.single_task([=]() {
    fp8_e5m2_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != std::numeric_limits<float>::infinity())
    ret = 1;
  if (out[1] != -std::numeric_limits<float>::infinity())
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_overflow_no_saturation(sycl::queue &queue) {
  const float input[2] = {60000.0f, -60000.0f};
  auto *data = sycl::malloc_shared<fp8_e5m2_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e5m2_x2(input, rounding::to_even, saturation::none);

  queue.single_task([=]() {
    fp8_e5m2_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != std::numeric_limits<float>::infinity())
    ret = 1;
  if (out[1] != -std::numeric_limits<float>::infinity())
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

template <typename T> int test_fp8_simple_type_conversion(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp8_e5m2_x2>(1, queue);
  data[0] = fp8_e5m2_x2(static_cast<T>(1.5f), static_cast<T>(2.5f));

  queue.single_task([=]() {
    fp8_e5m2_x2 value = data[0];
    sycl::marray<T, 2> f = static_cast<sycl::marray<T, 2>>(value);
    f[0] += static_cast<T>(1.0f);
    f[1] += static_cast<T>(1.0f);
    data[0] = fp8_e5m2_x2(f);
  });
  queue.wait_and_throw();

  sycl::marray<T, 2> expected_input(static_cast<T>(2.5f), static_cast<T>(3.5f));
  fp8_e5m2_x2 expected(expected_input);
  sycl::marray<T, 2> out = static_cast<sycl::marray<T, 2>>(data[0]);
  sycl::marray<T, 2> expected_out = static_cast<sycl::marray<T, 2>>(expected);

  sycl::free(data, queue);
  for (size_t i = 0; i < 2; ++i) {
    if (std::fabs(static_cast<float>(out[i]) -
                  static_cast<float>(expected_out[i])) > 0.0f)
      return 1;
  }

  return 0;
}

template <typename T> int test_marray_conversion(sycl::queue &queue) {
  sycl::marray<T, 2> input(static_cast<T>(1.0f), static_cast<T>(2.0f));
  auto *data = sycl::malloc_shared<fp8_e5m2_x2>(1, queue);
  data[0] = fp8_e5m2_x2(input);

  queue.single_task([=]() {
    fp8_e5m2_x2 value = data[0];
    sycl::marray<T, 2> f = static_cast<sycl::marray<T, 2>>(value);
    f[0] += static_cast<T>(1.0f);
    f[1] += static_cast<T>(2.0f);
    data[0] = fp8_e5m2_x2(f);
  });
  queue.wait_and_throw();
  sycl::marray<T, 2> expected_input(static_cast<T>(2.0f), static_cast<T>(4.0f));
  fp8_e5m2_x2 expected(expected_input);
  sycl::marray<T, 2> out = static_cast<sycl::marray<T, 2>>(data[0]);
  sycl::marray<T, 2> expected_out = static_cast<sycl::marray<T, 2>>(expected);

  sycl::free(data, queue);
  for (size_t i = 0; i < 2; ++i) {
    if (std::fabs(static_cast<float>(out[i]) -
                  static_cast<float>(expected_out[i])) > 0.0f)
      return 1;
  }
  return 0;
}

template <typename T> int test_carray_conversion(sycl::queue &queue) {
  T input[2] = {static_cast<T>(1.0f), static_cast<T>(3.0f)};
  auto *data = sycl::malloc_shared<fp8_e5m2_x2>(1, queue);
  data[0] = fp8_e5m2_x2(input);

  queue.single_task([=]() {
    fp8_e5m2_x2 value = data[0];
    sycl::marray<T, 2> unpacked = static_cast<sycl::marray<T, 2>>(value);
    T output[2] = {unpacked[0] + static_cast<T>(1.0f),
                   unpacked[1] + static_cast<T>(4.0f)};
    data[0] = fp8_e5m2_x2(output);
  });
  queue.wait_and_throw();

  T expected_input[2] = {static_cast<T>(2.0f), static_cast<T>(7.0f)};
  fp8_e5m2_x2 expected(expected_input);
  sycl::marray<T, 2> out = static_cast<sycl::marray<T, 2>>(data[0]);
  sycl::marray<T, 2> expected_out = static_cast<sycl::marray<T, 2>>(expected);

  sycl::free(data, queue);
  for (size_t i = 0; i < 2; ++i) {
    if (std::fabs(static_cast<float>(out[i]) -
                  static_cast<float>(expected_out[i])) > 0.0f)
      return 1;
  }

  return 0;
}

int main() {
  static_assert(alignof(fp8_e5m2_x2) == 2);
  static_assert(sizeof(fp8_e5m2_x2) == 2);

  auto async_handler = [](sycl::exception_list exceptions) {
    for (const std::exception_ptr &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (const sycl::exception &ex) {
        std::cerr << "Async SYCL exception: " << ex.what() << '\n';
        std::terminate();
      }
    }
  };

  sycl::queue queue{async_handler};

  int ret = test_fp8_simple_type_conversion<float>(queue);
  ret |= test_fp8_simple_type_conversion<sycl::half>(queue);
  ret |= test_fp8_simple_type_conversion<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_marray_conversion<float>(queue);
  ret |= test_marray_conversion<sycl::half>(queue);
  ret |= test_marray_conversion<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_carray_conversion<float>(queue);
  ret |= test_carray_conversion<sycl::half>(queue);
  ret |= test_carray_conversion<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_explicit_to_even_carray_constructor<float>(queue);
  ret |= test_explicit_to_even_carray_constructor<sycl::half>(queue);
  ret |= test_explicit_to_even_carray_constructor<sycl::ext::oneapi::bfloat16>(
      queue);

  ret |= test_explicit_to_even_marray_constructor<float>(queue);
  ret |= test_explicit_to_even_marray_constructor<sycl::half>(queue);
  ret |= test_explicit_to_even_marray_constructor<sycl::ext::oneapi::bfloat16>(
      queue);

  ret |= test_boundary_round_trip_nan(queue);
  ret |= test_boundary_round_trip_negative_zero(queue);
  ret |= test_boundary_round_trip_subnormals(queue);
  ret |= test_boundary_round_trip_exact_normals(queue);
  ret |= test_boundary_round_trip_exact_subnormal_limits(queue);
  ret |= test_boundary_round_trip_saturation_and_infinity_clamp(queue);
  ret |= test_boundary_infinity_no_saturation(queue);
  ret |= test_boundary_overflow_no_saturation(queue);

  ret |=
      test_stochastic_constructor<sycl::half, false, saturation::finite>(queue);
  ret |=
      test_stochastic_constructor<sycl::half, false, saturation::none>(queue);
  ret |=
      test_stochastic_constructor<sycl::half, true, saturation::finite>(queue);
  ret |= test_stochastic_constructor<sycl::half, true, saturation::none>(queue);
  ret |= test_stochastic_constructor<sycl::ext::oneapi::bfloat16, false,
                                     saturation::finite>(queue);
  ret |= test_stochastic_constructor<sycl::ext::oneapi::bfloat16, false,
                                     saturation::none>(queue);
  ret |= test_stochastic_constructor<sycl::ext::oneapi::bfloat16, true,
                                     saturation::finite>(queue);
  ret |= test_stochastic_constructor<sycl::ext::oneapi::bfloat16, true,
                                     saturation::none>(queue);
  return ret;
}
