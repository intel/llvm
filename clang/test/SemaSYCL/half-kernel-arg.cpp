// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test checks that compiler generates correct initialization for arguments
// that have sycl::half type inside the OpenCL kernel

#include "sycl.hpp"

sycl::queue myQueue;

int main() {
  sycl::half HostHalf;
  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_half>(
        [=]() {
          sycl::half KernelHalf = HostHalf;
        });
  });
}

// CHECK: {{.*}}kernel_half{{.*}} 'void (sycl::half)'
// CHECK: ParmVarDecl {{.*}} used _arg_HostHalf 'sycl::half':'sycl::detail::half_impl::half'
// // Check that lambda field of half type is initialized
// CHECK: VarDecl {{.*}} used __wrapper_union '__wrapper_union'
// CHECK:      CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .HostHalf
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} '_arg_HostHalf'
// CHECK-NEXT:  IntegerLiteral {{.*}} 2
