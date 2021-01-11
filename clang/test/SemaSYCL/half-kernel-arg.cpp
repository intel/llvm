// RUN: %clang_cc1 -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct initialization for arguments
// that have cl::sycl::half type inside the OpenCL kernel

#include "Inputs/sycl.hpp"

int main() {
  cl::sycl::half HostHalf;
  cl::sycl::kernel_single_task<class kernel_half>(
      [=]() {
        cl::sycl::half KernelHalf = HostHalf;
      });
}

// CHECK: {{.*}}kernel_half{{.*}} 'void (cl::sycl::half)'
// CHECK: ParmVarDecl {{.*}} used _arg_ 'cl::sycl::half':'cl::sycl::detail::half_impl::half'
// // Check that lambda field of half type is initialized
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}'
// CHECK-NEXT: InitListExpr {{.*}}'(lambda at {{.*}}'
// CHECK-NEXT: CXXConstructExpr {{.*}}'cl::sycl::detail::half_impl::half'{{.*}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const cl::sycl::detail::half_impl::half'
// CHECK-NEXT: DeclRefExpr {{.*}} 'cl::sycl::half':'cl::sycl::detail::half_impl::half' lvalue ParmVar {{.*}} '_arg_' 'cl::sycl::half':'cl::sycl::detail::half_impl::half'
