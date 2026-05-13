
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
  auto *data = sycl::malloc_shared<fp8_e5m2>(1, queue);
  data[0] = fp8_e5m2(static_cast<T>(1.5));

  queue.single_task([=]() {
    fp8_e5m2 value = data[0];
    T f = static_cast<T>(value);
    f += static_cast<T>(1.0f);
    data[0] = fp8_e5m2(f);
  });
  queue.wait_and_throw();

  fp8_e5m2 expected(2.5f);
  T out = static_cast<T>(data[0]);
  T expected_out = static_cast<T>(expected);

  sycl::free(data, queue);
  if (std::fabs(out - expected_out) > 0.0f)
    return 1;

  return 0;
}

int test_boolean_conversion(sycl::queue &queue, float test_value,
                            bool expected) {
  auto *data = sycl::malloc_shared<fp8_e5m2>(1, queue);
  auto *res = sycl::malloc_shared<bool>(1, queue);
  data[0] = fp8_e5m2(test_value);
  queue.single_task([=]() {
    fp8_e5m2 value = data[0];
    res[0] = static_cast<bool>(value);
  });
  queue.wait_and_throw();
  int ret = res[0] == expected ? 0 : 1;
  sycl::free(data, queue);
  sycl::free(res, queue);
  return ret;
}

template <typename T>
int test_single_element_carray_constructor(sycl::queue &queue) {
  T input[1] = {static_cast<T>(1.5f)};
  auto *data = sycl::malloc_shared<fp8_e5m2>(1, queue);
  data[0] = fp8_e5m2(input);

  queue.single_task([=]() {
    fp8_e5m2 value = data[0];
    T output[1] = {static_cast<T>(value) + static_cast<T>(1.0f)};
    data[0] = fp8_e5m2(output);
  });
  queue.wait_and_throw();

  fp8_e5m2 expected(static_cast<T>(2.5f));
  T out = static_cast<T>(data[0]);
  T expected_out = static_cast<T>(expected);

  sycl::free(data, queue);
  if (std::fabs(static_cast<float>(out) - static_cast<float>(expected_out)) >
      0.0f)
    return 1;
  return 0;
}

template <typename T> int test_marray_conversion(sycl::queue &queue) {
  sycl::marray<T, 1> input(static_cast<T>(1.5f));
  auto *data = sycl::malloc_shared<fp8_e5m2>(1, queue);
  data[0] = fp8_e5m2(input);

  queue.single_task([=]() {
    fp8_e5m2 value = data[0];
    sycl::marray<T, 1> f = static_cast<sycl::marray<T, 1>>(value);
    f[0] += static_cast<T>(1.0f);
    data[0] = fp8_e5m2(f);
  });
  queue.wait_and_throw();

  return 0;
}

template <typename T> int test_carray_conversion(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp8_e5m2>(1, queue);
  data[0] = fp8_e5m2(static_cast<T>(1.5f));

  queue.single_task([=]() {
    fp8_e5m2 value = data[0];
    T f = {static_cast<T>(value)};
    f += static_cast<T>(1.0f);
    data[0] = fp8_e5m2(f);
  });
  queue.wait_and_throw();

  fp8_e5m2 expected(static_cast<T>(2.5f));
  T out = {static_cast<T>(data[0])};
  T expected_out = {static_cast<T>(expected)};

  sycl::free(data, queue);
  if (std::fabs(out - expected_out) > 0.0f)
    return 1;
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

  ret |= test_boolean_conversion(queue, 0.0f, false);
  ret |= test_boolean_conversion(queue, -0.0f, false);
  ret |= test_boolean_conversion(queue, 1.0f, true);
  ret |= test_boolean_conversion(queue, -1.0f, true);
  ret |= test_boolean_conversion(queue, std::numeric_limits<float>::quiet_NaN(),
                                 true);
  ret |= test_boolean_conversion(queue, std::numeric_limits<float>::infinity(),
                                 true);
  ret |= test_boolean_conversion(queue, 1.52587890625e-05f, true);

  ret |= test_single_element_carray_constructor<float>(queue);
  ret |= test_single_element_carray_constructor<sycl::half>(queue);
  // ret |=
  //     test_single_element_carray_constructor<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_marray_conversion<float>(queue);
  ret |= test_marray_conversion<sycl::half>(queue);
  // TODO: uncomment when bfloat16 conversion is fixed
  // ret |= test_marray_conversion<sycl::ext::oneapi::bfloat16>(queue);

  ret |= test_carray_conversion<float>(queue);
  ret |= test_carray_conversion<sycl::half>(queue);
  // TODO: uncomment when bfloat16 conversion is fixed
  // ret |= test_carray_conversion<sycl::ext::oneapi::bfloat16>(queue);
  return ret;
}
