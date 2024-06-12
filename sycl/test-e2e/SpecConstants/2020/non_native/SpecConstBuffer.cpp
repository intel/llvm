// REQUIRES: opencl-aot, cpu

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 %S/Inputs/common.cpp -o %t.out \
// RUN:          -fsycl-dead-args-optimization
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out | FileCheck %s

#include <sycl/detail/core.hpp>

#include <sycl/specialization_id.hpp>

const static sycl::specialization_id<int> SpecConst{42};

int main() {
  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    CGH.set_specialization_constant<SpecConst>(1);
    CGH.single_task<class KernelName>([=](sycl::kernel_handler KH) {
      (void)KH.get_specialization_constant<SpecConst>();
    });
  });
  Q.wait();
  return 0;
  // CHECK: piMemRelease
}
