// RUN: %clang_cc1 -S -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test checks if the compiler correctly initializes the SYCL Sampler object when passed as a kernel argument.

#include "sycl.hpp"

sycl::queue myQueue;

int main() {

  sycl::sampler Sampler;

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class SamplerLambda>([=] {
      Sampler.use();
    });
  });

  return 0;
}

// Check declaration of the test kernel
// CHECK: FunctionDecl {{.*}}SamplerLambda{{.*}} 'void (sampler_t)'
//
// Check parameters of the test kernel
// CHECK: ParmVarDecl {{.*}} used [[_arg_sampler:[0-9a-zA-Z_]+]] 'sampler_t'
//
// Check that sampler field of the test kernel object is initialized using __init method
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__ocl_sampler_t)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::sampler':'sycl::sampler' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}sampler.cpp{{.*}})' lvalue Var {{.*}} '(lambda at {{.*}}sampler.cpp{{.*}})'
//
// Check the parameters of __init method
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__ocl_sampler_t':'sampler_t' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'sampler_t' lvalue ParmVar {{.*}} '[[_arg_sampler]]' 'sampler_t'
