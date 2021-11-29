// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s
// This test checks that Clang doesn't crash if a specialization constant is
// value dependent.

#include "sycl.hpp"
sycl::queue myQueue;

int main() {
   constexpr int default_val = 20;
   cl::sycl::ext::oneapi::experimental::spec_constant<int, class MyInt32Const> SC(default_val);
  
  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_sc>(
        [=] {
          cl::sycl::ext::oneapi::experimental::spec_constant<int, class MyInt32Const> res = SC;
        });
  });
  return 0;
}

// CHECK: FunctionDecl {{.*}}kernel_sc{{.*}} 'void ()'
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}'
// CHECK-NEXT: InitListExpr {{.*}}'(lambda at {{.*}}'
// CHECK-NEXT: CXXConstructExpr {{.*}}'cl::sycl::ext::oneapi::experimental::spec_constant<int, class MyInt32Const>':'sycl::ext::oneapi::experimental::spec_constant<int, MyInt32Const>' 'void ()'
