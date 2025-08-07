// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -internal-isystem %S/Inputs -fsycl-is-device -std=c++20 -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

#include "sycl.hpp"

// This test checks that we correctly capture binding declarations.

void foo() {
  sycl::handler h;
  int a[2] = {1, 2};
  auto [x, y] = a;
  struct S {
    float b[3] = { 0, 3.0f, 4.0 };
  } s;
  auto [f1, f2, f3] = s.b;
  auto Lambda = [=]() { (void)x; (void)f2; };
  h.single_task(Lambda);
}

// CHECK: %class.anon = type { i32, float }

// Check the sycl kernel arguments - one int and one float parameter
// CHECK: define {{.*}} spir_kernel void @{{.*}}foov{{.*}}(ptr noundef byval(%class.anon) align 4 %_arg__sycl_functor)
// CHECK: entry:
// CHECK: %_arg__sycl_functor.ascast = addrspacecast ptr %_arg__sycl_functor to ptr addrspace(4)

// Call the lambda
// CHECK: call spir_func void @{{.*}}foo{{.*}}(ptr addrspace(4) {{.*}} %_arg__sycl_functor.ascast)
// CHECK:   ret void

// Check the lambda call
// CHECK: define {{.*}} spir_func void @{{.*}}foo{{.*}}(ptr addrspace(4) {{.*}} %this)
// CHECK: entry:
// CHECK:  %this.addr = alloca ptr addrspace(4)
// CHECK:  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
// CHECK:  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast
// CHECK:  %this1 = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast

// Check the fetch of the x binding.
// CHECK:  %x = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %this1, i32 0, i32 0

// Check the fetch of the f2 binding.
// CHECK:  %f2 = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %this1, i32 0, i32 1
// CHECK:  ret void
