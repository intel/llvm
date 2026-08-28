// REQUIRES: arch-intel_gpu_cri
// RUN: %{build} -Xclang -freg-struct-return -Xspirv-translator=spir64 --spirv-ext=+SPV_INTEL_fp_conversions,+SPV_INTEL_float4,+SPV_KHR_bfloat16 -o %t.out
// RUN: %{run} SYCL_UR_TRACE=1 %t.out

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

template <typename T> int test_fp4_simple_type_conversion(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp4_e2m1>(1, queue);
  data[0] = fp4_e2m1(static_cast<T>(1.5));

  queue.single_task([=]() {
    fp4_e2m1 value = data[0];
    T f = static_cast<T>(value);
    f += static_cast<T>(1.0f);
    data[0] = fp4_e2m1(f);
  });
  queue.wait_and_throw();

  // E2M1 representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives).
  // 1.5 + 1.0 = 2.5 -> rounds to either 2 (to_even) or 3.
  // We compare via an fp4 round-trip so the expected is well-defined.
  fp4_e2m1 expected(
      static_cast<T>(static_cast<float>(static_cast<T>(1.5)) + 1.0f));
  T out = static_cast<T>(data[0]);
  T expected_out = static_cast<T>(expected);

  sycl::free(data, queue);
  if (std::fabs(out - expected_out) > 0.0f)
    return 1;

  return 0;
}

int test_boolean_conversion(sycl::queue &queue, float test_value,
                            bool expected) {
  auto *data = sycl::malloc_shared<fp4_e2m1>(1, queue);
  auto *res = sycl::malloc_shared<bool>(1, queue);
  data[0] = fp4_e2m1(test_value);
  queue.single_task([=]() {
    fp4_e2m1 value = data[0];
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
  auto *data = sycl::malloc_shared<fp4_e2m1>(1, queue);
  data[0] = fp4_e2m1(input);

  queue.single_task([=]() {
    fp4_e2m1 value = data[0];
    T output[1] = {static_cast<T>(value) + static_cast<T>(1.0f)};
    data[0] = fp4_e2m1(output);
  });
  queue.wait_and_throw();

  // 1.5 + 1.0 = 2.5; the closest representable values are 2.0 and 3.0,
  // round-to-even resolves the tie to 2.0 (frac=0).
  fp4_e2m1 expected(static_cast<T>(2.0f));
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
  auto *data = sycl::malloc_shared<fp4_e2m1>(1, queue);
  data[0] = fp4_e2m1(input);

  queue.single_task([=]() {
    fp4_e2m1 value = data[0];
    sycl::marray<T, 1> f = static_cast<sycl::marray<T, 1>>(value);
    f[0] += static_cast<T>(1.0f);
    data[0] = fp4_e2m1(f);
  });
  queue.wait_and_throw();

  sycl::marray<T, 1> expected_input(static_cast<T>(2.0f));
  fp4_e2m1 expected(expected_input);
  sycl::marray<T, 1> out = static_cast<sycl::marray<T, 1>>(data[0]);
  sycl::marray<T, 1> expected_out = static_cast<sycl::marray<T, 1>>(expected);

  sycl::free(data, queue);
  if (std::fabs(out[0] - expected_out[0]) > 0.0f)
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

  int ret = test_fp4_simple_type_conversion<float>(queue);
  ret |= test_fp4_simple_type_conversion<sycl::half>(queue);
  ret |= test_fp4_simple_type_conversion<sycl::ext::oneapi::bfloat16>(queue);
  ret |= test_fp4_simple_type_conversion<short>(queue);
  ret |= test_fp4_simple_type_conversion<unsigned short>(queue);
  ret |= test_fp4_simple_type_conversion<int>(queue);
  ret |= test_fp4_simple_type_conversion<unsigned int>(queue);
  ret |= test_fp4_simple_type_conversion<long>(queue);
  ret |= test_fp4_simple_type_conversion<unsigned long>(queue);
  ret |= test_fp4_simple_type_conversion<long long>(queue);
  ret |= test_fp4_simple_type_conversion<unsigned long long>(queue);
  ret |= test_fp4_simple_type_conversion<char>(queue);
  ret |= test_fp4_simple_type_conversion<signed char>(queue);
  ret |= test_fp4_simple_type_conversion<unsigned char>(queue);

  // bool conversion: only +0/-0 -> false; everything else -> true (E2M1 has
  // no NaN representation).
  ret |= test_boolean_conversion(queue, 0.0f, false);
  ret |= test_boolean_conversion(queue, -0.0f, false);
  ret |= test_boolean_conversion(queue, 1.0f, true);
  ret |= test_boolean_conversion(queue, -1.0f, true);
  ret |= test_boolean_conversion(queue, 0.5f, true);

  ret |= test_single_element_carray_constructor<float>(queue);
  ret |= test_single_element_carray_constructor<sycl::half>(queue);
  ret |= test_single_element_carray_constructor<sycl::ext::oneapi::bfloat16>(
      queue);

  ret |= test_marray_conversion<float>(queue);
  ret |= test_marray_conversion<sycl::half>(queue);
  ret |= test_marray_conversion<sycl::ext::oneapi::bfloat16>(queue);
  return ret;
}
