// REQUIRES: arch-intel_gpu_cri
// RUN: %{build} -Xclang -freg-struct-return -Xspirv-translator=spir64 --spirv-ext=+SPV_INTEL_fp_conversions,+SPV_INTEL_float4,+SPV_KHR_bfloat16 -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: target-nvidia, target-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with CRI driver, and the
// SPIR-V backend does not support the required SPIR-V extensions

#include <iostream>

#include <cmath>
#include <limits>
#include <sycl/ext/oneapi/experimental/float_4bit/types.hpp>
#include <sycl/queue.hpp>
#include <sycl/usm.hpp>

using namespace sycl::ext::oneapi::experimental;

bool equal_with_zero_sign(float actual, float expected) {
  if (actual != expected)
    return false;
  if (expected == 0.0f)
    return std::signbit(actual) == std::signbit(expected);
  return true;
}

template <typename T>
int test_explicit_to_even_carray_constructor(sycl::queue &queue) {
  // E2M1 exactly representable: 0.5 and -3.0 (positive subnormal & normal).
  T input[2] = {static_cast<T>(0.5f), static_cast<T>(-3.0f)};
  auto *data = sycl::malloc_shared<fp4_e2m1_x2>(1, queue);
  auto *out = sycl::malloc_shared<T>(2, queue);
  data[0] = fp4_e2m1_x2(input, rounding::to_even);

  queue.single_task([=]() {
    sycl::marray<T, 2> unpacked = static_cast<sycl::marray<T, 2>>(data[0]);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (static_cast<float>(out[0]) != 0.5f)
    ret = 1;
  if (static_cast<float>(out[1]) != -3.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

template <typename T>
int test_explicit_to_even_marray_constructor(sycl::queue &queue) {
  sycl::marray<T, 2> input(static_cast<T>(1.5f), static_cast<T>(-4.0f));
  auto *data = sycl::malloc_shared<fp4_e2m1_x2>(1, queue);
  auto *out = sycl::malloc_shared<T>(2, queue);
  data[0] = fp4_e2m1_x2(input, rounding::to_even);

  queue.single_task([=]() {
    sycl::marray<T, 2> unpacked = static_cast<sycl::marray<T, 2>>(data[0]);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (static_cast<float>(out[0]) != 1.5f)
    ret = 1;
  if (static_cast<float>(out[1]) != -4.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_round_trip_negative_zero(sycl::queue &queue) {
  const float input[2] = {-0.0f, 6.0f};
  auto *data = sycl::malloc_shared<fp4_e2m1_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp4_e2m1_x2(input, rounding::to_even);

  queue.single_task([=]() {
    fp4_e2m1_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp4_e2m1_x2(unpacked, rounding::to_even);
    sycl::marray<float, 2> round_tripped =
        static_cast<sycl::marray<float, 2>>(data[0]);
    out[0] = round_tripped[0];
    out[1] = round_tripped[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (!equal_with_zero_sign(out[0], -0.0f))
    ret = 1;
  if (out[1] != 6.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_round_trip_subnormals(sycl::queue &queue) {
  // The only positive subnormal is 0.5; -0.5 is the only negative subnormal.
  const float input[2] = {0.5f, -0.5f};
  auto *data = sycl::malloc_shared<fp4_e2m1_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp4_e2m1_x2(input, rounding::to_even);

  queue.single_task([=]() {
    fp4_e2m1_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp4_e2m1_x2(unpacked, rounding::to_even);
    sycl::marray<float, 2> round_tripped =
        static_cast<sycl::marray<float, 2>>(data[0]);
    out[0] = round_tripped[0];
    out[1] = round_tripped[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != 0.5f)
    ret = 1;
  if (out[1] != -0.5f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_round_trip_exact_normals(sycl::queue &queue) {
  const float input[2] = {6.0f, 1.0f};
  auto *data = sycl::malloc_shared<fp4_e2m1_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp4_e2m1_x2(input, rounding::to_even);

  queue.single_task([=]() {
    fp4_e2m1_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp4_e2m1_x2(unpacked, rounding::to_even);
    sycl::marray<float, 2> round_tripped =
        static_cast<sycl::marray<float, 2>>(data[0]);
    out[0] = round_tripped[0];
    out[1] = round_tripped[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != 6.0f)
    ret = 1;
  if (out[1] != 1.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_round_trip_saturation_and_infinity_clamp(sycl::queue &queue) {
  // 100.0 saturates to +6.0; -inf saturates to -6.0 (no Inf in E2M1).
  const float input[2] = {100.0f, -std::numeric_limits<float>::infinity()};
  auto *data = sycl::malloc_shared<fp4_e2m1_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp4_e2m1_x2(input, rounding::to_even);

  queue.single_task([=]() {
    fp4_e2m1_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp4_e2m1_x2(unpacked, rounding::to_even);
    sycl::marray<float, 2> round_tripped =
        static_cast<sycl::marray<float, 2>>(data[0]);
    out[0] = round_tripped[0];
    out[1] = round_tripped[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != 6.0f)
    ret = 1;
  if (out[1] != -6.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

template <typename T> int test_fp4_simple_type_conversion(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp4_e2m1_x2>(1, queue);
  data[0] = fp4_e2m1_x2(static_cast<T>(1.5f), static_cast<T>(2.0f));

  queue.single_task([=]() {
    fp4_e2m1_x2 value = data[0];
    sycl::marray<T, 2> f = static_cast<sycl::marray<T, 2>>(value);
    f[0] += static_cast<T>(1.5f); // 1.5 + 1.5 = 3.0 (exactly representable)
    f[1] += static_cast<T>(2.0f); // 2.0 + 2.0 = 4.0 (exactly representable)
    data[0] = fp4_e2m1_x2(f);
  });
  queue.wait_and_throw();

  sycl::marray<T, 2> expected_input(static_cast<T>(3.0f), static_cast<T>(4.0f));
  fp4_e2m1_x2 expected(expected_input);
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
  auto *data = sycl::malloc_shared<fp4_e2m1_x2>(1, queue);
  data[0] = fp4_e2m1_x2(input);

  queue.single_task([=]() {
    fp4_e2m1_x2 value = data[0];
    sycl::marray<T, 2> f = static_cast<sycl::marray<T, 2>>(value);
    f[0] += static_cast<T>(2.0f); // 1+2=3 (exact)
    f[1] += static_cast<T>(2.0f); // 2+2=4 (exact)
    data[0] = fp4_e2m1_x2(f);
  });
  queue.wait_and_throw();
  sycl::marray<T, 2> expected_input(static_cast<T>(3.0f), static_cast<T>(4.0f));
  fp4_e2m1_x2 expected(expected_input);
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
  T input[2] = {static_cast<T>(1.0f), static_cast<T>(2.0f)};
  auto *data = sycl::malloc_shared<fp4_e2m1_x2>(1, queue);
  data[0] = fp4_e2m1_x2(input);

  queue.single_task([=]() {
    fp4_e2m1_x2 value = data[0];
    sycl::marray<T, 2> unpacked = static_cast<sycl::marray<T, 2>>(value);
    T output[2] = {unpacked[0] + static_cast<T>(2.0f),
                   unpacked[1] + static_cast<T>(2.0f)};
    data[0] = fp4_e2m1_x2(output);
  });
  queue.wait_and_throw();

  T expected_input[2] = {static_cast<T>(3.0f), static_cast<T>(4.0f)};
  fp4_e2m1_x2 expected(expected_input);
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

  // fp4_e2m1_x2 only supports packed conversions through marray<half, 2>,
  // marray<bfloat16, 2>, and marray<float, 2>.
  int ret = test_fp4_simple_type_conversion<float>(queue);
  ret |= test_fp4_simple_type_conversion<sycl::half>(queue);
  ret |= test_fp4_simple_type_conversion<sycl::ext::oneapi::bfloat16>(queue);

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

  ret |= test_boundary_round_trip_negative_zero(queue);
  ret |= test_boundary_round_trip_subnormals(queue);
  ret |= test_boundary_round_trip_exact_normals(queue);
  ret |= test_boundary_round_trip_saturation_and_infinity_clamp(queue);
  return ret;
}
