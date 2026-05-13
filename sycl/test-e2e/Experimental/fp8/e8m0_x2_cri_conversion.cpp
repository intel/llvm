
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

template <typename T>
int test_explicit_upward_carray_constructor(sycl::queue &queue) {
  T input[2] = {static_cast<T>(4.0f), static_cast<T>(16.0f)};
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  auto *out = sycl::malloc_shared<T>(2, queue);
  data[0] = fp8_e8m0_x2(input, rounding::upward);

  queue.single_task([=]() {
    sycl::marray<T, 2> unpacked = static_cast<sycl::marray<T, 2>>(data[0]);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (static_cast<float>(out[0]) != 4.0f)
    ret = 1;
  if (static_cast<float>(out[1]) != 16.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

template <typename T>
int test_explicit_toward_zero_carray_constructor(sycl::queue &queue) {
  T input[2] = {static_cast<T>(5.0f), static_cast<T>(12.0f)};
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  auto *out = sycl::malloc_shared<T>(2, queue);
  data[0] = fp8_e8m0_x2(input, rounding::toward_zero);

  queue.single_task([=]() {
    sycl::marray<T, 2> unpacked = static_cast<sycl::marray<T, 2>>(data[0]);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (static_cast<float>(out[0]) != 4.0f)
    ret = 1;
  if (static_cast<float>(out[1]) != 8.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

template <typename T>
int test_explicit_upward_marray_constructor(sycl::queue &queue) {
  sycl::marray<T, 2> input(static_cast<T>(2.0f), static_cast<T>(64.0f));
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  auto *out = sycl::malloc_shared<T>(2, queue);
  data[0] = fp8_e8m0_x2(input, rounding::upward);

  queue.single_task([=]() {
    sycl::marray<T, 2> unpacked = static_cast<sycl::marray<T, 2>>(data[0]);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (static_cast<float>(out[0]) != 2.0f)
    ret = 1;
  if (static_cast<float>(out[1]) != 64.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

template <typename T>
int test_explicit_toward_zero_marray_constructor(sycl::queue &queue) {
  sycl::marray<T, 2> input(static_cast<T>(3.0f), static_cast<T>(10.0f));
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  auto *out = sycl::malloc_shared<T>(2, queue);
  data[0] = fp8_e8m0_x2(input, rounding::toward_zero);

  queue.single_task([=]() {
    sycl::marray<T, 2> unpacked = static_cast<sycl::marray<T, 2>>(data[0]);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (static_cast<float>(out[0]) != 2.0f)
    ret = 1;
  if (static_cast<float>(out[1]) != 8.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_round_trip_nan(sycl::queue &queue) {
  const float input[2] = {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::quiet_NaN()};
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e8m0_x2(input, rounding::upward);

  queue.single_task([=]() {
    fp8_e8m0_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp8_e8m0_x2(unpacked, rounding::upward);
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

int test_boundary_round_trip_exact_powers_of_two(sycl::queue &queue) {
  const float input[2] = {32.0f, 0.25f};
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e8m0_x2(input, rounding::upward);

  queue.single_task([=]() {
    fp8_e8m0_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp8_e8m0_x2(unpacked, rounding::upward);
    sycl::marray<float, 2> round_tripped =
        static_cast<sycl::marray<float, 2>>(data[0]);
    out[0] = round_tripped[0];
    out[1] = round_tripped[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != 32.0f)
    ret = 1;
  if (out[1] != 0.25f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_round_trip_max_min_normal(sycl::queue &queue) {
  float max_val = std::ldexp(1.0f, 127);
  float min_val = std::ldexp(1.0f, -127);
  const float input[2] = {max_val, min_val};
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e8m0_x2(input, rounding::toward_zero);

  queue.single_task([=]() {
    fp8_e8m0_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    data[0] = fp8_e8m0_x2(unpacked, rounding::toward_zero);
    sycl::marray<float, 2> round_tripped =
        static_cast<sycl::marray<float, 2>>(data[0]);
    out[0] = round_tripped[0];
    out[1] = round_tripped[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != max_val)
    ret = 1;
  if (out[1] != min_val)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_saturation_infinity_clamp(sycl::queue &queue) {
  const float input[2] = {std::numeric_limits<float>::infinity(),
                          std::ldexp(1.0f, 128)};
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e8m0_x2(input, rounding::upward);

  queue.single_task([=]() {
    fp8_e8m0_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  float max_e8m0 = std::ldexp(1.0f, 127);
  int ret = 0;
  if (out[0] != max_e8m0)
    ret = 1;
  if (out[1] != max_e8m0)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_boundary_negative_input_drops_sign(sycl::queue &queue) {
  const float input[2] = {-4.0f, -32.0f};
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e8m0_x2(input, rounding::upward);

  queue.single_task([=]() {
    fp8_e8m0_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != 4.0f)
    ret = 1;
  if (out[1] != 32.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_rounding_upward_non_power_of_two(sycl::queue &queue) {
  const float input[2] = {3.0f, 6.0f};
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e8m0_x2(input, rounding::upward);

  queue.single_task([=]() {
    fp8_e8m0_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != 4.0f)
    ret = 1;
  if (out[1] != 8.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_rounding_toward_zero_non_power_of_two(sycl::queue &queue) {
  const float input[2] = {3.0f, 6.0f};
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  auto *out = sycl::malloc_shared<float>(2, queue);
  data[0] = fp8_e8m0_x2(input, rounding::toward_zero);

  queue.single_task([=]() {
    fp8_e8m0_x2 value = data[0];
    sycl::marray<float, 2> unpacked =
        static_cast<sycl::marray<float, 2>>(value);
    out[0] = unpacked[0];
    out[1] = unpacked[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != 2.0f)
    ret = 1;
  if (out[1] != 4.0f)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_raw_vals_access(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  auto *out = sycl::malloc_shared<uint8_t>(2, queue);
  float input[2] = {1.0f, 2.0f};
  data[0] = fp8_e8m0_x2(input, rounding::upward);

  queue.single_task([=]() {
    out[0] = data[0].vals[0];
    out[1] = data[0].vals[1];
  });
  queue.wait_and_throw();

  int ret = 0;
  if (out[0] != 127)
    ret = 1;
  if (out[1] != 128)
    ret = 1;

  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

} // namespace

template <typename T> int test_fp8_simple_type_conversion(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  data[0] = fp8_e8m0_x2(static_cast<T>(4.0f), static_cast<T>(16.0f));

  queue.single_task([=]() {
    fp8_e8m0_x2 value = data[0];
    sycl::marray<T, 2> f = static_cast<sycl::marray<T, 2>>(value);
    f[0] *= static_cast<T>(2.0f);
    f[1] *= static_cast<T>(2.0f);
    data[0] = fp8_e8m0_x2(f);
  });
  queue.wait_and_throw();

  sycl::marray<T, 2> expected_input(static_cast<T>(8.0f),
                                    static_cast<T>(32.0f));
  fp8_e8m0_x2 expected(expected_input);
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
  sycl::marray<T, 2> input(static_cast<T>(4.0f), static_cast<T>(16.0f));
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  data[0] = fp8_e8m0_x2(input);

  queue.single_task([=]() {
    fp8_e8m0_x2 value = data[0];
    sycl::marray<T, 2> f = static_cast<sycl::marray<T, 2>>(value);
    f[0] *= static_cast<T>(2.0f);
    f[1] *= static_cast<T>(4.0f);
    data[0] = fp8_e8m0_x2(f);
  });
  queue.wait_and_throw();
  sycl::marray<T, 2> expected_input(static_cast<T>(8.0f),
                                    static_cast<T>(64.0f));
  fp8_e8m0_x2 expected(expected_input);
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
  T input[2] = {static_cast<T>(4.0f), static_cast<T>(16.0f)};
  auto *data = sycl::malloc_shared<fp8_e8m0_x2>(1, queue);
  data[0] = fp8_e8m0_x2(input);

  queue.single_task([=]() {
    fp8_e8m0_x2 value = data[0];
    sycl::marray<T, 2> unpacked = static_cast<sycl::marray<T, 2>>(value);
    T output[2] = {unpacked[0] * static_cast<T>(2.0f),
                   unpacked[1] * static_cast<T>(4.0f)};
    data[0] = fp8_e8m0_x2(output);
  });
  queue.wait_and_throw();

  T expected_input[2] = {static_cast<T>(8.0f), static_cast<T>(64.0f)};
  fp8_e8m0_x2 expected(expected_input);
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

  int ret = test_fp8_simple_type_conversion<float>(queue);
  ret |= test_fp8_simple_type_conversion<sycl::half>(queue);
  ret |= test_fp8_simple_type_conversion<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_marray_conversion<float>(queue);
  ret |= test_marray_conversion<sycl::half>(queue);
  ret |= test_marray_conversion<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_carray_conversion<float>(queue);
  ret |= test_carray_conversion<sycl::half>(queue);
  ret |= test_carray_conversion<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_explicit_upward_carray_constructor<float>(queue);
  ret |= test_explicit_upward_carray_constructor<sycl::half>(queue);
  ret |= test_explicit_upward_carray_constructor<sycl::ext::oneapi::bfloat16>(
      queue);

  ret |= test_explicit_toward_zero_carray_constructor<float>(queue);
  ret |= test_explicit_toward_zero_carray_constructor<sycl::half>(queue);
  ret |=
      test_explicit_toward_zero_carray_constructor<sycl::ext::oneapi::bfloat16>(
          queue);

  ret |= test_explicit_upward_marray_constructor<float>(queue);
  ret |= test_explicit_upward_marray_constructor<sycl::half>(queue);
  ret |= test_explicit_upward_marray_constructor<sycl::ext::oneapi::bfloat16>(
      queue);

  ret |= test_explicit_toward_zero_marray_constructor<float>(queue);
  ret |= test_explicit_toward_zero_marray_constructor<sycl::half>(queue);
  ret |=
      test_explicit_toward_zero_marray_constructor<sycl::ext::oneapi::bfloat16>(
          queue);

  ret |= test_boundary_round_trip_nan(queue);
  ret |= test_boundary_round_trip_exact_powers_of_two(queue);
  ret |= test_boundary_round_trip_max_min_normal(queue);
  ret |= test_boundary_saturation_infinity_clamp(queue);
  ret |= test_boundary_negative_input_drops_sign(queue);
  ret |= test_rounding_upward_non_power_of_two(queue);
  ret |= test_rounding_toward_zero_non_power_of_two(queue);
  ret |= test_raw_vals_access(queue);
  return ret;
}
