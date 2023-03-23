// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test checks that compiler generates correct initialization for spec
// constants

#include "sycl.hpp"

sycl::queue myQueue;

struct SpecConstantsWrapper {
  sycl::ext::oneapi::experimental::spec_constant<int, class sc_name1> SC1;
  sycl::ext::oneapi::experimental::spec_constant<int, class sc_name2> SC2;
};

int main() {
  sycl::ext::oneapi::experimental::spec_constant<char, class MyInt32Const> SC;
  SpecConstantsWrapper SCWrapper;
  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_sc>(
        [=] {
          (void)SC;
          (void)SCWrapper;
        });
  });
}

// CHECK: FunctionDecl {{.*}}kernel_sc{{.*}} 'void ()'
// CHECK: VarDecl {{.*}} used __wrapper_union '__wrapper_union'
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr {{.*}} 'sycl::ext::oneapi::experimental::spec_constant<char, MyInt32Const>'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .SC
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT: CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr {{.*}} 'sycl::ext::oneapi::experimental::spec_constant<int, sc_name1>'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .SC1
// CHECK-NEXT:     MemberExpr {{.*}} .SCWrapper
// CHECK-NEXT:      MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:       DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT: CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr {{.*}} 'sycl::ext::oneapi::experimental::spec_constant<int, sc_name2>'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .SC2
// CHECK-NEXT:     MemberExpr {{.*}} .SCWrapper
// CHECK-NEXT:      MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:       DeclRefExpr {{.*}} '__wrapper_union'
