// REQUIRES: gpu

// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s
//
// TODO: Reenable on Windows, see https://github.com/intel/llvm/issues/14768
// XFAIL: hip_nvidia, windows

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

// CHECK:---> urContextRelease(
