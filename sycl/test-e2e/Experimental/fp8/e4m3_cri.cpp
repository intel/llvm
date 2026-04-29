// RUN: %{build} -Xclang -freg-struct-return -Xspirv-translator --spirv-ext=+SPV_INTEL_fp_conversions,+SPV_EXT_float8,+SPV_KHR_bfloat16 -o %t.out
// RUN: %{run} SYCL_UR_TRACE=1 %t.out

// Warning! This test requires CRI simulator run to communicate via TCP socket
// with port 60999, or any other from config

#include <cmath>
#include <iostream>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>
#include <sycl/queue.hpp>
#include <sycl/usm.hpp>

using namespace sycl::ext::oneapi::experimental;

template <typename T> int run_basic_fp8_test(sycl::queue &queue) {
  auto *data = sycl::malloc_shared<fp8_e4m3>(1, queue);
  data[0] = fp8_e4m3(static_cast<T>(1.5));

  queue.single_task([=]() {
    fp8_e4m3 value = data[0];
    T f = static_cast<T>(value);
    f += 1.0f;
    data[0] = fp8_e4m3(f);
  });
  queue.wait_and_throw();

  fp8_e4m3 expected(2.5f);
  T out = static_cast<T>(data[0]);
  T expected_out = static_cast<T>(expected);

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

  int ret = run_basic_fp8_test<float>(queue);
  ret |= run_basic_fp8_test<double>(queue);
  ret |= run_basic_fp8_test<sycl::half>(queue);
  ret |= run_basic_fp8_test<sycl::ext::oneapi::bfloat16>(queue);

  return ret;
}
