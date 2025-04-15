// REQUIRES: opencl-aot, cpu

// CPU AOT targets host isa, so we compile on the run system instead.
// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 %S/Inputs/common.cpp -o %t.out \
// RUN:          -fsycl-dead-args-optimization
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s

#include <sycl/detail/core.hpp>

#include <sycl/kernel_bundle.hpp>
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
  // CHECK: <--- urMemRelease
}
