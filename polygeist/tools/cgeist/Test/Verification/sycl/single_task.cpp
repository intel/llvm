// clang-format off
// RUN: clang++ -fsycl -fsycl-device-only -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o %t.bc
// Test that the LLVMIR generated is verifiable.
// RUN: opt -verify -disable-output < %t.bc
// Verify that LLVMIR generated is translatable to SPIRV.
// RUN: llvm-spirv %t.bc
// RUN: llvm-dis %t.bc
// RUN: cat %t.ll | FileCheck %s
// BUG: clang++ -o doesn't redirect the output to the file specified.
// XFAIL: *

// Test that the kernel named `kernel_single_task` is generated with the correct signature.
// CHECK: define weak_odr spir_kernel void {{.*}}kernel_single_task(i32 addrspace(1)* {{.*}}, [[RANGE_TY:%"class.sycl::_V1::range.1"]]* byval([[RANGE_TY]]) {{.*}}, [[RANGE_TY]]* byval([[RANGE_TY]]) {{.*}}, [[ID_TY:%"class.sycl::_V1::id.1"]]* byval([[ID_TY]]) {{.*}})

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
