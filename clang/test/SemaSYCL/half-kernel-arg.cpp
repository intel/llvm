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
// CHECK: ParmVarDecl {{.*}} used _arg_ 'sycl::half':'sycl::detail::half_impl::half'
// // Check that lambda field of half type is initialized
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}'
// CHECK-NEXT: InitListExpr {{.*}}'(lambda at {{.*}}'
// CHECK-NEXT: CXXConstructExpr {{.*}}'sycl::detail::half_impl::half'{{.*}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const sycl::detail::half_impl::half'
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::half':'sycl::detail::half_impl::half' lvalue ParmVar {{.*}} '_arg_' 'sycl::half':'sycl::detail::half_impl::half'
