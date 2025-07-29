// REQUIRES: gpu

// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out %if !windows %{2>&1 | FileCheck %s %}

#include <sycl/detail/core.hpp>

int main() {
  const auto GlobalRange = 1;
  const auto LocalRange = 2;

  sycl::queue myQueue{sycl::gpu_selector_v, [](sycl::exception_list elist) {
                        for (auto e : elist)
                          std::rethrow_exception(e);
                      }};

  try {
    // Generating an exception caused by the fact that LocalRange size (== 2)
    // can't be greater than GlobalRange size (== 1)
    myQueue.parallel_for<class TestKernel>(
        sycl::nd_range<1>{sycl::range<1>(GlobalRange),
                          sycl::range<1>(LocalRange)},
        [=](sycl::nd_item<1> idx) {});
    myQueue.wait_and_throw();
    assert(false && "Expected exception was not caught");
  } catch (sycl::exception &e) {
  }

  return 0;
}

// On Windows, dlls unloading is inconsistent and if we try to release these UR
// objects manually, inconsistent hangs happen due to a race between unloading
// the UR adapters dlls (in addition to their dependency dlls) and the releasing
// of these UR objects. So, we currently shutdown without releasing them and
// windows should handle the memory cleanup.

// CHECK: <--- urContextRelease(
