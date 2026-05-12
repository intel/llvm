
// RUN: %{build} -Xclang -freg-struct-return -Xspirv-translator --spirv-ext=+SPV_INTEL_fp_conversions,+SPV_EXT_float8,+SPV_KHR_bfloat16 -o %t.out
// RUN: %{run} SYCL_UR_TRACE=1 %t.out

// Warning! This test requires CRI device or its simulator run to communicate
// via TCP socket with port 60999, or any other from config

// TODO need to set requirement of intel_feature_gpu_cri

#include <cmath>
#include <limits>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>
#include <sycl/queue.hpp>
#include <sycl/usm.hpp>

using namespace sycl::ext::oneapi::experimental;

namespace {

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

template <typename T>
int test_explicit_to_even_carray_constructor(sycl::queue &queue) {
  T input[2] = {static_cast<T>(0.01171875f), static_cast<T>(-5.5f)};
  auto *data = sycl::malloc_shared<fp8_e4m3_x2>(1, queue);
  auto *out = sycl::malloc_shared<T>(2, queue);
  data[0] = fp8_e4m3_x2(input, rounding::to_even);

  queue.single_task([=]() {
    sycl::marray<T, 2> unpacked = static_cast<sycl::marray<T, 2>>(data[0]);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (static_cast<float>(out[0]) != 0.01171875f)
    ret = 1;
  if (static_cast<float>(out[1]) != -5.5f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

template <typename T>
int test_explicit_to_even_marray_constructor(sycl::queue &queue) {
  sycl::marray<T, 2> input(static_cast<T>(3.25f),
                           static_cast<T>(-0.009765625f));
  auto *data = sycl::malloc_shared<fp8_e4m3_x2>(1, queue);
  auto *out = sycl::malloc_shared<T>(2, queue);
  data[0] = fp8_e4m3_x2(input, rounding::to_even);

  queue.single_task([=]() {
    sycl::marray<T, 2> unpacked = static_cast<sycl::marray<T, 2>>(data[0]);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (static_cast<float>(out[0]) != 3.25f)
    ret = 1;
  if (static_cast<float>(out[1]) != -0.009765625f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_round_trip_nan(sycl::queue &queue) {
  const float input[2] = {std::numeric_limits<float>::quiet_NaN(),
                          -std::numeric_limits<float>::quiet_NaN()};
  auto *data = sycl::malloc_shared<fp8_e4m3_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e4m3_x2(input, rounding::to_even);

  queue.single_task([=]() {
    fp8_e4m3_x2 value = data[0];
    sycl::marray<float, 2> unpacked = static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp8_e4m3_x2(unpacked, rounding::to_even);
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
  auto *data = sycl::malloc_shared<fp8_e4m3_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e4m3_x2(input, rounding::to_even);

  queue.single_task([=]() {
    fp8_e4m3_x2 value = data[0];
    sycl::marray<float, 2> unpacked = static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp8_e4m3_x2(unpacked, rounding::to_even);
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
  const float input[2] = {0.01171875f, -0.009765625f};
  auto *data = sycl::malloc_shared<fp8_e4m3_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e4m3_x2(input, rounding::to_even);

  queue.single_task([=]() {
    fp8_e4m3_x2 value = data[0];
    sycl::marray<float, 2> unpacked = static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp8_e4m3_x2(unpacked, rounding::to_even);
    sycl::marray<float, 2> round_tripped =
        static_cast<sycl::marray<float, 2>>(data[0]);
    out[0] = round_tripped[0];
    out[1] = round_tripped[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != 0.01171875f)
    ret = 1;
  if (out[1] != -0.009765625f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_round_trip_saturation_and_infinity_clamp(sycl::queue &queue) {
  const float input[2] = {600.0f, -std::numeric_limits<float>::infinity()};
  auto *data = sycl::malloc_shared<fp8_e4m3_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e4m3_x2(input, rounding::to_even);

  queue.single_task([=]() {
    fp8_e4m3_x2 value = data[0];
    sycl::marray<float, 2> unpacked = static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp8_e4m3_x2(unpacked, rounding::to_even);
    sycl::marray<float, 2> round_tripped =
        static_cast<sycl::marray<float, 2>>(data[0]);
    out[0] = round_tripped[0];
    out[1] = round_tripped[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != 448.0f)
    ret = 1;
  if (out[1] != -448.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

} // namespace

template <typename T> int test_fp8_simple_type_conversion(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp8_e4m3_x2>(1, queue);
  data[0] = fp8_e4m3_x2(static_cast<T>(1.5f), static_cast<T>(2.5f));

  queue.single_task([=]() {
    fp8_e4m3_x2 value = data[0];
    sycl::marray<T, 2> f = static_cast<sycl::marray<T, 2>>(value);
    f[0] += static_cast<T>(1.0f);
    f[1] += static_cast<T>(1.0f);
    data[0] = fp8_e4m3_x2(f);
  });
  queue.wait_and_throw();

  sycl::marray<T, 2> expected_input(static_cast<T>(2.5f), static_cast<T>(3.5f));
  fp8_e4m3_x2 expected(expected_input);
  sycl::marray<T, 2> out = static_cast<sycl::marray<T, 2>>(data[0]);
  sycl::marray<T, 2> expected_out = static_cast<sycl::marray<T, 2>>(expected);

  sycl::free(data, queue);
  for (size_t i = 0; i < 2; ++i) {
    if (std::fabs(static_cast<float>(out[i]) - static_cast<float>(expected_out[i])) >
        0.0f)
      return 1;
  }

  return 0;
}

template <typename T> int test_marray_conversion(sycl::queue &queue) {
  sycl::marray<T, 2> input(static_cast<T>(1.25f), static_cast<T>(2.5f));
  auto *data = sycl::malloc_shared<fp8_e4m3_x2>(1, queue);
  data[0] = fp8_e4m3_x2(input);

  queue.single_task([=]() {
    fp8_e4m3_x2 value = data[0];
    sycl::marray<T, 2> f = static_cast<sycl::marray<T, 2>>(value);
    f[0] += static_cast<T>(1.0f);
    f[1] += static_cast<T>(2.0f);
    data[0] = fp8_e4m3_x2(f);
  });
  queue.wait_and_throw();
  sycl::marray<T, 2> expected_input(static_cast<T>(2.25f), static_cast<T>(4.5f));
  fp8_e4m3_x2 expected(expected_input);
  sycl::marray<T, 2> out = static_cast<sycl::marray<T, 2>>(data[0]);
  sycl::marray<T, 2> expected_out = static_cast<sycl::marray<T, 2>>(expected);

  sycl::free(data, queue);
  for (size_t i = 0; i < 2; ++i) {
    if (std::fabs(static_cast<float>(out[i]) - static_cast<float>(expected_out[i])) >
        0.0f)
      return 1;
  }
  return 0;
}

template <typename T> int test_carray_conversion(sycl::queue &queue) {
  T input[2] = {static_cast<T>(1.25f), static_cast<T>(2.5f)};
  auto *data = sycl::malloc_shared<fp8_e4m3_x2>(1, queue);
  data[0] = fp8_e4m3_x2(input);

  queue.single_task([=]() {
    fp8_e4m3_x2 value = data[0];
    sycl::marray<T, 2> unpacked = static_cast<sycl::marray<T, 2>>(value);
    T output[2] = {unpacked[0] + static_cast<T>(1.0f),
                   unpacked[1] + static_cast<T>(4.0f)};
    data[0] = fp8_e4m3_x2(output);
  });
  queue.wait_and_throw();

  T expected_input[2] = {static_cast<T>(2.25f), static_cast<T>(6.5f)};
  fp8_e4m3_x2 expected(expected_input);
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

  // fp8_e4m3_x2 only supports packed conversions through marray<half, 2>,
  // marray<bfloat16, 2>, and marray<float, 2>.
  int ret = test_fp8_simple_type_conversion<float>(queue);
  ret |= test_fp8_simple_type_conversion<sycl::half>(queue);
  ret |= test_fp8_simple_type_conversion<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_marray_conversion<float>(queue);
  ret |= test_marray_conversion<sycl::half>(queue);
//  ret |= test_marray_conversion<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_carray_conversion<float>(queue);
  ret |= test_carray_conversion<sycl::half>(queue);
 // ret |= test_carray_conversion<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_explicit_to_even_carray_constructor<float>(queue);
  ret |= test_explicit_to_even_carray_constructor<sycl::half>(queue);
 // ret |= test_explicit_to_even_carray_constructor<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_explicit_to_even_marray_constructor<float>(queue);
  ret |= test_explicit_to_even_marray_constructor<sycl::half>(queue);
 // ret |= test_explicit_to_even_marray_constructor<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_boundary_round_trip_nan(queue);
  ret |= test_boundary_round_trip_negative_zero(queue);
  ret |= test_boundary_round_trip_subnormals(queue);
  ret |= test_boundary_round_trip_saturation_and_infinity_clamp(queue);
  return ret;
}
