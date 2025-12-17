// REQUIRES-INTEL-DRIVER: cpu: 2026
// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/clock.hpp>
#include <sycl/usm.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

template <syclex::clock_scope scope> void test(sycl::queue &q) {
  auto *data = sycl::malloc_shared<uint64_t>(3, q);

  q.parallel_for(2, [=](sycl::id<1> idx) {
     if (idx == 0) {
       data[0] = syclex::clock<scope>();
       int sum = 0;
       for (int i = 0; i < 1'000'000; ++i)
         sum += i;
       data[1] = syclex::clock<scope>();
       sum = 0;
       for (int i = 0; i < 1'000'000; ++i)
         sum += i;
       data[2] = syclex::clock<scope>();
     }
   }).wait();

  assert(data[1] > data[0]);
  assert(data[2] > data[1]);
  sycl::free(data, q);
}

template <syclex::clock_scope scope>
void test_if_supported(sycl::queue &q, sycl::aspect asp) {
  auto dev = q.get_device();
  if (dev.has(asp))
    test<scope>(q);
  else
    try {
      test<scope>(q);
    } catch (sycl::exception &e) {
      assert(e.code() == sycl::errc::kernel_not_supported && "Unexpected errc");
    }
}

int main() {
  sycl::queue q;
  test_if_supported<syclex::clock_scope::sub_group>(
      q, sycl::aspect::ext_oneapi_clock_sub_group);
  test_if_supported<syclex::clock_scope::work_group>(
      q, sycl::aspect::ext_oneapi_clock_work_group);
  test_if_supported<syclex::clock_scope::device>(
      q, sycl::aspect::ext_oneapi_clock_device);
  return 0;
}
