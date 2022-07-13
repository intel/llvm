// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %ACC_RUN_PLACEHOLDER %t.out 2>&1 %ACC_CHECK_PLACEHOLDER

// JIT compilation path is not taken on CUDA and HIP backend.
// UNSUPPORTED: cuda || hip

// This tests checks that implicitly created kernel_bundles (i.e. through
// setting a specialization ID from host) only builds the device image
// containing the kernel it launches.

#include <sycl/sycl.hpp>

#include <iostream>

class Kernel1;
class Kernel2;

constexpr sycl::specialization_id<int> SpecConst;

int main() {
  sycl::queue Q;

  int Ret = 0;
  {
    sycl::buffer<int, 1> Buf(&Ret, 1);
    Q.submit([&](sycl::handler &CGH) {
      auto Acc = Buf.template get_access<sycl::access::mode::write>(CGH);
      CGH.set_specialization_constant<SpecConst>(42);
      CGH.single_task<class Kernel1>([=](sycl::kernel_handler H) {
        Acc[0] = H.get_specialization_constant<SpecConst>();
      });
    });
    Q.wait_and_throw();
  }

  if (Ret == 1) {
    // This should never happen but we need the kernel
    Q.submit(
        [&](sycl::handler &CGH) { CGH.single_task<class Kernel2>([=]() {}); });
    Q.wait_and_throw();
  }
  std::cout << "passed" << std::endl;
  return 0;
}

// --- Check that only a single program is built:
// CHECK: ---> piProgramBuild
// CHECK-NOT: ---> piProgramBuild
// --- Check that the test completed with expected results:
// CHECK: passed
