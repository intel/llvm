// RUN: sycl-clang.py %s -S -emit-llvm -o %t.ll
// Test that the LLVMIR generated is verifiable.
// RUN: opt -verify -disable-output < %t.ll
// Verify that LLVMIR generated is translatable to SPIRV.
// RUN: llvm-as %t.ll
// RUN: llvm-spirv %t.bc
// RUN: cat %t.ll | FileCheck %s
// XFAIL: *

// Test that the kernel named `kernel_single_task` is generated.
// CHECK: define weak_odr spir_kernel void {{.*}}kernel_single_task
// Test that all referenced sycl header functions are generated.
// CHECK-NOT: declare {{.*}} spir_func

#include <sycl/sycl.hpp>

void host_single_task() {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{1};

  {
    auto buf = sycl::buffer<int, 1>{nullptr, range};
    q.submit([&](sycl::handler &cgh) {
      auto A = buf.get_access<sycl::access::mode::write>(cgh);
      cgh.single_task<class kernel_single_task>([=]() {
        A[0] = 42;
      });
    });
  }
}
