// REQUIRES: usm_shared_allocations
// For this test, complete_fusion must be supported.
// RUN: %{build} -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 %{run} %t.out 2>&1 | FileCheck %s

// Test fusion cancellation on event dependency between two active fusions.

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  constexpr size_t dataSize = 512;

  queue q1{ext::codeplay::experimental::property::queue::enable_fusion{}};
  queue q2{ext::codeplay::experimental::property::queue::enable_fusion{}};

  int *in1 = sycl::malloc_shared<int>(dataSize, q1);
  int *in2 = sycl::malloc_shared<int>(dataSize, q1);
  int *in3 = sycl::malloc_shared<int>(dataSize, q1);
  int *tmp = sycl::malloc_shared<int>(dataSize, q1);
  int *out = sycl::malloc_shared<int>(dataSize, q1);

  for (size_t i = 0; i < dataSize; ++i) {
    in1[i] = i * 2;
    in2[i] = i * 3;
    in3[i] = i * 4;
    tmp[i] = -1;
    out[i] = -1;
  }

  ext::codeplay::experimental::fusion_wrapper fw1{q1};
  fw1.start_fusion();

  assert(fw1.is_in_fusion_mode() && "Queue should be in fusion mode");

  auto kernel1 = q1.submit([&](handler &cgh) {
    cgh.parallel_for<class KernelOne>(
        dataSize, [=](id<1> i) { tmp[i] = in1[i] + in2[i]; });
  });

  ext::codeplay::experimental::fusion_wrapper fw2{q2};
  fw2.start_fusion();

  auto kernel3 = q2.submit([&](handler &cgh) {
    cgh.depends_on(kernel1);
    cgh.parallel_for<class KernelThree>(dataSize,
                                        [=](id<1> i) { tmp[i] *= 2; });
  });

  // kernel3 specifies an event dependency on kernel1. To avoid circular
  // dependencies between two fusions, the fusion for q1 needs to cancelled.
  assert(!fw1.is_in_fusion_mode() &&
         "Queue should not be in fusion mode anymore");

  assert(fw2.is_in_fusion_mode() && "Queue should be in fusion mode");

  auto kernel2 = q1.submit([&](handler &cgh) {
    cgh.depends_on(kernel3);
    cgh.parallel_for<class KernelTwo>(
        dataSize, [=](id<1> i) { out[i] = tmp[i] * in3[i]; });
  });

  // kernel2 specifies an event dependency on kernel3, which leads to
  // cancellation of the fusion for q2.
  assert(!fw2.is_in_fusion_mode() &&
         "Queue should not be in fusion mode anymore");

  fw1.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

  fw2.cancel_fusion();

  q1.wait();
  q2.wait();

  // Check the results
  for (size_t i = 0; i < dataSize; ++i) {
    assert(out[i] == (40 * i * i) && "Computation error");
  }
  sycl::free(in1, q1);
  sycl::free(in2, q1);
  sycl::free(in3, q1);
  sycl::free(tmp, q1);
  sycl::free(out, q1);
  return 0;
}

// CHECK: WARNING: Aborting fusion because of event dependency from a different fusion
// CHECK-NEXT: WARNING: Aborting fusion because synchronization with one of the kernels in the fusion list was requested
