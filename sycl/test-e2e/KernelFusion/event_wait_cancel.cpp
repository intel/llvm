// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: %{run} %t.out

// Test validity of events after cancel_fusion.

#include "fusion_event_test_common.h"

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

  auto kernel1 = q.submit([&](handler &cgh) {
    cgh.parallel_for<class KernelOne>(
        dataSize, [=](id<1> i) { tmp[i] = in1[i] + in2[i]; });
  });

  auto kernel2 = q.submit([&](handler &cgh) {
    cgh.depends_on(kernel1);
    cgh.parallel_for<class KernelTwo>(
        dataSize, [=](id<1> i) { out[i] = tmp[i] * in3[i]; });
  });

  fw.cancel_fusion();

  assert(!fw.is_in_fusion_mode() &&
         "Queue should not be in fusion mode anymore");

  kernel1.wait();
  assert(isEventComplete(kernel1) && "Event should be complete");
  // The event returned by submit while in fusion mode depends on both
  // individual kernels to be executed.
  assert(kernel1.get_wait_list().size() == 2);

  kernel2.wait();
  assert(isEventComplete(kernel2) && "Event should be complete");
  // The event returned by submit while in fusion mode depends on both
  // individual kernels to be executed.
  assert(kernel2.get_wait_list().size() == 2);

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
