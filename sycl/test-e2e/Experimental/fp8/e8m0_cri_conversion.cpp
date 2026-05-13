
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

template <typename T> int test_fp8_simple_type_conversion(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  data[0] = fp8_e8m0(static_cast<T>(4.0f));

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    T f = static_cast<T>(value);
    f *= static_cast<T>(2.0f);
    data[0] = fp8_e8m0(f);
  });
  queue.wait_and_throw();

  fp8_e8m0 expected(8.0f);
  T out = static_cast<T>(data[0]);
  T expected_out = static_cast<T>(expected);

  sycl::free(data, queue);
  if (std::fabs(static_cast<float>(out) - static_cast<float>(expected_out)) >
      0.0f)
    return 1;

  return 0;
}

int test_boolean_conversion(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  auto *res = sycl::malloc_shared<bool>(1, queue);
  data[0] = fp8_e8m0(1.0f);
  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    res[0] = static_cast<bool>(value);
  });
  queue.wait_and_throw();
  int ret = res[0] == true ? 0 : 1;
  sycl::free(data, queue);
  sycl::free(res, queue);
  return ret;
}

int test_boolean_conversion_large(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  auto *res = sycl::malloc_shared<bool>(1, queue);
  data[0] = fp8_e8m0(128.0f);
  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    res[0] = static_cast<bool>(value);
  });
  queue.wait_and_throw();
  int ret = res[0] == true ? 0 : 1;
  sycl::free(data, queue);
  sycl::free(res, queue);
  return ret;
}

int test_boolean_conversion_nan(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  auto *res = sycl::malloc_shared<bool>(1, queue);
  data[0] = fp8_e8m0(std::numeric_limits<float>::quiet_NaN());
  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    res[0] = static_cast<bool>(value);
  });
  queue.wait_and_throw();
  int ret = res[0] == true ? 0 : 1;
  sycl::free(data, queue);
  sycl::free(res, queue);
  return ret;
}

template <typename T>
int test_single_element_carray_constructor(sycl::queue &queue) {
  T input[1] = {static_cast<T>(4.0f)};
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  data[0] = fp8_e8m0(input);

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    T output[1] = {static_cast<T>(value) * static_cast<T>(2.0f)};
    data[0] = fp8_e8m0(output);
  });
  queue.wait_and_throw();

  fp8_e8m0 expected(static_cast<T>(8.0f));
  T out = static_cast<T>(data[0]);
  T expected_out = static_cast<T>(expected);

  sycl::free(data, queue);
  if (std::fabs(static_cast<float>(out) - static_cast<float>(expected_out)) >
      0.0f)
    return 1;
  return 0;
}

template <typename T> int test_marray_conversion(sycl::queue &queue) {
  sycl::marray<T, 1> input(static_cast<T>(4.0f));
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  data[0] = fp8_e8m0(input);

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    sycl::marray<T, 1> f = static_cast<sycl::marray<T, 1>>(value);
    f[0] *= static_cast<T>(2.0f);
    data[0] = fp8_e8m0(f);
  });
  queue.wait_and_throw();

  sycl::marray<T, 1> expected_input(static_cast<T>(8.0f));
  fp8_e8m0 expected(expected_input);
  sycl::marray<T, 1> out = static_cast<sycl::marray<T, 1>>(data[0]);
  sycl::marray<T, 1> expected_out = static_cast<sycl::marray<T, 1>>(expected);

  sycl::free(data, queue);
  if (std::fabs(static_cast<float>(out[0]) -
                static_cast<float>(expected_out[0])) > 0.0f)
    return 1;
  return 0;
}

template <typename T> int test_carray_conversion(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  data[0] = fp8_e8m0(static_cast<T>(4.0f));

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    T f = {static_cast<T>(value)};
    f *= static_cast<T>(2.0f);
    data[0] = fp8_e8m0(f);
  });
  queue.wait_and_throw();

  fp8_e8m0 expected(static_cast<T>(8.0f));
  T out = {static_cast<T>(data[0])};
  T expected_out = {static_cast<T>(expected)};

  sycl::free(data, queue);
  if (std::fabs(static_cast<float>(out) - static_cast<float>(expected_out)) >
      0.0f)
    return 1;
  return 0;
}

int test_rounding_upward(sycl::queue &queue) {
  float input[1] = {3.0f};
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  data[0] = fp8_e8m0(input, rounding::upward);

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    float out = static_cast<float>(value);
    float expected[1] = {out};
    data[0] = fp8_e8m0(expected, rounding::upward);
  });
  queue.wait_and_throw();

  float out = static_cast<float>(data[0]);
  sycl::free(data, queue);
  if (out != 4.0f)
    return 1;
  return 0;
}

int test_rounding_toward_zero(sycl::queue &queue) {
  float input[1] = {3.0f};
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  data[0] = fp8_e8m0(input, rounding::toward_zero);

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    float out = static_cast<float>(value);
    float expected[1] = {out};
    data[0] = fp8_e8m0(expected, rounding::toward_zero);
  });
  queue.wait_and_throw();

  float out = static_cast<float>(data[0]);
  sycl::free(data, queue);
  if (out != 2.0f)
    return 1;
  return 0;
}

int test_rounding_upward_marray(sycl::queue &queue) {
  sycl::marray<float, 1> input(5.0f);
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  data[0] = fp8_e8m0(input, rounding::upward);

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    sycl::marray<float, 1> f = static_cast<sycl::marray<float, 1>>(value);
    data[0] = fp8_e8m0(f, rounding::upward);
  });
  queue.wait_and_throw();

  float out = static_cast<float>(data[0]);
  sycl::free(data, queue);
  if (out != 8.0f)
    return 1;
  return 0;
}

int test_rounding_toward_zero_marray(sycl::queue &queue) {
  sycl::marray<float, 1> input(5.0f);
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  data[0] = fp8_e8m0(input, rounding::toward_zero);

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    sycl::marray<float, 1> f = static_cast<sycl::marray<float, 1>>(value);
    data[0] = fp8_e8m0(f, rounding::toward_zero);
  });
  queue.wait_and_throw();

  float out = static_cast<float>(data[0]);
  sycl::free(data, queue);
  if (out != 4.0f)
    return 1;
  return 0;
}

int test_power_of_two_round_trip(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  auto *out = sycl::malloc_shared<float>(1, queue);
  data[0] = fp8_e8m0(16.0f);

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    out[0] = static_cast<float>(value);
  });
  queue.wait_and_throw();

  int ret = (out[0] != 16.0f) ? 1 : 0;
  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_max_normal_round_trip(sycl::queue &queue) {
  float max_val = std::ldexp(1.0f, 127);
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  auto *out = sycl::malloc_shared<float>(1, queue);
  float input[1] = {max_val};
  data[0] = fp8_e8m0(input, rounding::toward_zero);

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    out[0] = static_cast<float>(value);
  });
  queue.wait_and_throw();

  int ret = (out[0] != max_val) ? 1 : 0;
  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_min_normal_round_trip(sycl::queue &queue) {
  float min_val = std::ldexp(1.0f, -127);
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  auto *out = sycl::malloc_shared<float>(1, queue);
  float input[1] = {min_val};
  data[0] = fp8_e8m0(input, rounding::toward_zero);

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    out[0] = static_cast<float>(value);
  });
  queue.wait_and_throw();

  int ret = (out[0] != min_val) ? 1 : 0;
  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_nan_round_trip(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  auto *out = sycl::malloc_shared<float>(1, queue);
  float input[1] = {std::numeric_limits<float>::quiet_NaN()};
  data[0] = fp8_e8m0(input, rounding::upward);

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    out[0] = static_cast<float>(value);
  });
  queue.wait_and_throw();

  int ret = std::isnan(out[0]) ? 0 : 1;
  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_saturation_large_value(sycl::queue &queue) {
  float large = std::numeric_limits<float>::infinity();
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  auto *out = sycl::malloc_shared<float>(1, queue);
  float input[1] = {large};
  data[0] = fp8_e8m0(input, rounding::upward);

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    out[0] = static_cast<float>(value);
  });
  queue.wait_and_throw();

  float max_e8m0 = std::ldexp(1.0f, 127);
  int ret = (out[0] != max_e8m0) ? 1 : 0;
  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_saturation_overflow(sycl::queue &queue) {
  float large = std::ldexp(1.0f, 128);
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  auto *out = sycl::malloc_shared<float>(1, queue);
  float input[1] = {large};
  data[0] = fp8_e8m0(input, rounding::upward);

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    out[0] = static_cast<float>(value);
  });
  queue.wait_and_throw();

  float max_e8m0 = std::ldexp(1.0f, 127);
  int ret = (out[0] != max_e8m0) ? 1 : 0;
  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_raw_vals_access(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  auto *out = sycl::malloc_shared<uint8_t>(1, queue);
  data[0] = fp8_e8m0(1.0f);

  queue.single_task([=]() {
    out[0] = data[0].vals[0];
  });
  queue.wait_and_throw();

  int ret = (out[0] != 127) ? 1 : 0;
  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
}

int test_negative_input_drops_sign(sycl::queue &queue) {
  float input[1] = {-8.0f};
  auto *data = sycl::malloc_shared<fp8_e8m0>(1, queue);
  auto *out = sycl::malloc_shared<float>(1, queue);
  data[0] = fp8_e8m0(input, rounding::upward);

  queue.single_task([=]() {
    fp8_e8m0 value = data[0];
    out[0] = static_cast<float>(value);
  });
  queue.wait_and_throw();

  int ret = (out[0] != 8.0f) ? 1 : 0;
  sycl::free(data, queue);
  sycl::free(out, queue);
  return ret;
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
  ret |= test_fp8_simple_type_conversion<short>(queue);
  ret |= test_fp8_simple_type_conversion<unsigned short>(queue);
  ret |= test_fp8_simple_type_conversion<int>(queue);
  ret |= test_fp8_simple_type_conversion<unsigned int>(queue);
  ret |= test_fp8_simple_type_conversion<long>(queue);
  ret |= test_fp8_simple_type_conversion<unsigned long>(queue);
  ret |= test_fp8_simple_type_conversion<long long>(queue);
  ret |= test_fp8_simple_type_conversion<unsigned long long>(queue);
  ret |= test_fp8_simple_type_conversion<char>(queue);
  ret |= test_fp8_simple_type_conversion<signed char>(queue);
  ret |= test_fp8_simple_type_conversion<unsigned char>(queue);

  ret |= test_boolean_conversion(queue);
  ret |= test_boolean_conversion_large(queue);
  ret |= test_boolean_conversion_nan(queue);

  ret |= test_single_element_carray_constructor<float>(queue);
  ret |= test_single_element_carray_constructor<sycl::half>(queue);
  ret |= test_single_element_carray_constructor<sycl::ext::oneapi::bfloat16>(
      queue);

  ret |= test_marray_conversion<float>(queue);
  ret |= test_marray_conversion<sycl::half>(queue);
  ret |= test_marray_conversion<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_carray_conversion<float>(queue);
  ret |= test_carray_conversion<sycl::half>(queue);
  ret |= test_carray_conversion<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_rounding_upward(queue);
  ret |= test_rounding_toward_zero(queue);
  ret |= test_rounding_upward_marray(queue);
  ret |= test_rounding_toward_zero_marray(queue);

  ret |= test_power_of_two_round_trip(queue);
  ret |= test_max_normal_round_trip(queue);
  ret |= test_min_normal_round_trip(queue);
  ret |= test_nan_round_trip(queue);
  ret |= test_saturation_large_value(queue);
  ret |= test_saturation_overflow(queue);
  ret |= test_raw_vals_access(queue);
  ret |= test_negative_input_drops_sign(queue);
  return ret;
}
