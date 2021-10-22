// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s
// This test checks that Clang doesn't crash if a specialization constant is
// value dependent.

#include "sycl.hpp"
sycl::queue myQueue;

int main() {
   constexpr int default_val = 20;
   sycl::ext::oneapi::experimental::spec_constant<int, class MyInt32Const> SC(default_val);
  
  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_sc>(
        [=] {
          sycl::ext::oneapi::experimental::spec_constant<int, class MyInt32Const> res = SC;
        });
  });
  return 0;
}

// CHECK: FunctionDecl {{.*}}kernel_sc{{.*}} 'void ()'
// CHECK: VarDecl {{.*}} __wrapper_union
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr {{.*}} 'sycl::ext::oneapi::experimental::spec_constant<int, MyInt32Const>' 'void ()'
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     MemberExpr {{.*}} .SC
// CHECK-NEXT:      MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:       DeclRefExpr {{.*}} '__wrapper_union'
