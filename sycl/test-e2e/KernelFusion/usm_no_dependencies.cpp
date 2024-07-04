// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} %{embed-ir} -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out | FileCheck %s

// Test complete fusion using USM pointers.

// The two kernels are fused, so only a single, fused kernel is launched.
// CHECK-COUNT-1: piEnqueueKernelLaunch
// CHECK-NOT: piEnqueueKernelLaunch

#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

int main() {
  constexpr size_t dataSize = 512;

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  int *in1 = sycl::malloc_shared<int>(dataSize, q);
  int *in2 = sycl::malloc_shared<int>(dataSize, q);
  int *in3 = sycl::malloc_shared<int>(dataSize, q);
  int *tmp = sycl::malloc_shared<int>(dataSize, q);
  int *out = sycl::malloc_shared<int>(dataSize, q);

  for (size_t i = 0; i < dataSize; ++i) {
    in1[i] = i * 2;
    in2[i] = i * 3;
    in3[i] = i * 4;
    tmp[i] = -1;
    out[i] = -1;
  }

  ext::codeplay::experimental::fusion_wrapper fw{q};
  fw.start_fusion();

  assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

  q.submit([&](handler &cgh) {
    cgh.parallel_for<class KernelOne>(
        dataSize, [=](id<1> i) { tmp[i] = in1[i] + in2[i]; });
  });

  q.submit([&](handler &cgh) {
    cgh.parallel_for<class KernelTwo>(
        dataSize, [=](id<1> i) { out[i] = tmp[i] * in3[i]; });
  });

  fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

  assert(!fw.is_in_fusion_mode() &&
         "Queue should not be in fusion mode anymore");

  q.wait();

  // Check the results
  for (size_t i = 0; i < dataSize; ++i) {
    assert(out[i] == (20 * i * i) && "Computation error");
  }

  sycl::free(in1, q);
  sycl::free(in2, q);
  sycl::free(in3, q);
  sycl::free(tmp, q);
  sycl::free(out, q);

  return 0;
}
