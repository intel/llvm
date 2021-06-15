// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test checks that compiler generates correct initialization for spec
// constants

#include "sycl.hpp"

sycl::queue myQueue;

struct SpecConstantsWrapper {
  sycl::ONEAPI::experimental::spec_constant<int, class sc_name1> SC1;
  sycl::ONEAPI::experimental::spec_constant<int, class sc_name2> SC2;
};

int main() {
  sycl::ONEAPI::experimental::spec_constant<char, class MyInt32Const> SC;
  SpecConstantsWrapper SCWrapper;
  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_sc>(
        [=] {
          (void)SC;
          (void)SCWrapper;
        });
  });
}

// CHECK: FunctionDecl {{.*}}kernel_sc{{.*}} 'void (sycl::ONEAPI::experimental::spec_constant<char, class MyInt32Const>, sycl::ONEAPI::experimental::spec_constant<int, class sc_name1>, sycl::ONEAPI::experimental::spec_constant<int, class sc_name2>)'
// CHECK: VarDecl {{.*}}'(lambda at {{.*}})'
// CHECK: InitListExpr {{.*}}'(lambda at {{.*}})'
// CHECK-NEXT: CXXConstructExpr {{.*}}'sycl::ONEAPI::experimental::spec_constant<char, class MyInt32Const>':'sycl::ONEAPI::experimental::spec_constant<char, MyInt32Const>'
// CHECK: InitListExpr {{.*}} 'SpecConstantsWrapper'
// CHECK: CXXConstructExpr {{.*}} 'sycl::ONEAPI::experimental::spec_constant<int, class sc_name1>':'sycl::ONEAPI::experimental::spec_constant<int, sc_name1>'
// CHECK: CXXConstructExpr {{.*}} 'sycl::ONEAPI::experimental::spec_constant<int, class sc_name2>':'sycl::ONEAPI::experimental::spec_constant<int, sc_name2>'
