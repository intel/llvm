// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

#include "sycl.hpp"

void take_bool(bool) {}

int main() {
  bool test = false;
  sycl::queue q;

  // CHECK: @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E11test_kernel(i8 {{.*}} [[ARG:%[A-Za-z_0-9]*]]
  // CHECK: %__SYCLKernel = alloca
  // CHECK: %test = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %__SYCLKernel.ascast
  // CHECK: store i8 %{{.*}}, ptr addrspace(4) %test
  // CHECK: call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv
  //
  // CHECK: define {{.*}} @_Z9take_boolb(i1
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel>([=]() {
      (void)test;
      take_bool(test);
    });
  });

  return 0;
}
