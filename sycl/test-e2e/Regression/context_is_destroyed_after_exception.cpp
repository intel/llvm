// REQUIRES: gpu

// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
//
// XFAIL: hip_nvidia

#include <sycl/sycl.hpp>

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

// CHECK:---> piContextRelease(
