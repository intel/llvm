// REQUIRES: opencl || level_zero, gpu
// UNSUPPORTED: gpu-intel-dg1 || windows
//
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device dg1" %s -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=2 %{run} %t.out 2>&1 | FileCheck %s

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q{sycl::default_selector_v};

  auto *I = sycl::malloc_device<int>(1, Q);
  Q.single_task([=]() { I[0] = 42; });

  sycl::free(I, Q);

  return 0;
}

// CHECK: The program was built for {{.*}} devices
