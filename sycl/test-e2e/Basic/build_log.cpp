// REQUIRES: opencl || level_zero, gpu, ocloc
// UNSUPPORTED: arch-intel_gpu_dg2
// UNSUPPORTED-INTENDED: see https://github.com/intel/llvm/pull/20643

// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg2 %s -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=2 %{run} %t.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;

  auto *I = sycl::malloc_device<int>(1, Q);
  Q.single_task([=]() { I[0] = 42; });

  sycl::free(I, Q);

  return 0;
}

// CHECK: The program was built for {{.*}} devices
