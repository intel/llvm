// REQUIRES: opencl || level_zero, gpu, ocloc, target-spir
// UNSUPPORTED: arch-intel_gpu_dg2
// UNSUPPORTED-INTENDED: see https://github.com/intel/llvm/pull/20643
// UNSUPPORTED: new-offload-model
// UNSUPPORTED-INTENDED: runs a kernel built for DG2 on another GPU to get a
// build log. intel_gpu_dg2 images are only used on DG2, and spir64_gen is an
// error with --offload-new-driver.

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device dg2" -Wno-deprecated %s -o %t.out
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
