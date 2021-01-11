// RUN: %clang_cc1 -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct initialization for spec
// constants

#include "Inputs/sycl.hpp"

struct SpecConstantsWrapper {
  cl::sycl::ONEAPI::experimental::spec_constant<int, class sc_name1> SC1;
  cl::sycl::ONEAPI::experimental::spec_constant<int, class sc_name2> SC2;
};

int main() {
  cl::sycl::ONEAPI::experimental::spec_constant<char, class MyInt32Const> SC;
  SpecConstantsWrapper W;
  cl::sycl::kernel_single_task<class kernel_sc>(
      [=]() {
        (void)SC;
        (void)W;
      });
}

// CHECK: FunctionDecl {{.*}}kernel_sc{{.*}} 'void ()'
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}'
// CHECK-NEXT: InitListExpr {{.*}}'(lambda at {{.*}}'
// CHECK-NEXT: CXXConstructExpr {{.*}}'cl::sycl::ONEAPI::experimental::spec_constant<char, class MyInt32Const>':'cl::sycl::ONEAPI::experimental::spec_constant<char, MyInt32Const>'
// CHECK-NEXT: InitListExpr {{.*}} 'SpecConstantsWrapper'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'cl::sycl::ONEAPI::experimental::spec_constant<int, class sc_name1>':'cl::sycl::ONEAPI::experimental::spec_constant<int, sc_name1>'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'cl::sycl::ONEAPI::experimental::spec_constant<int, class sc_name2>':'cl::sycl::ONEAPI::experimental::spec_constant<int, sc_name2>'
