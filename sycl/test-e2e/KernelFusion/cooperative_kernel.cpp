// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=2 %{run} %t.out 2>&1 | FileCheck %s

// Test cooperative kernels are not fused

// This test currently fails on AMD HIP due to an unresolved problem when
// submitting a kernel with the use_root_sync property to a command group with
// fusion enabled.
// XFAIL: hip_amd

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};
  ext::codeplay::experimental::fusion_wrapper fw(q);

  {
    // CHECK: Not fusing kernel with 'use_root_sync' property. Can only fuse non-cooperative device kernels.
    fw.start_fusion();
    q.submit([&](handler &cgh) {
      const auto props = sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::use_root_sync};
      cgh.parallel_for(sycl::range<1>{1}, props, [=](sycl::id<1>) {});
    });
    fw.complete_fusion();
  }

}
