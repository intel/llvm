// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -ast-dump %s | FileCheck %s

// Test for AST of reqd_work_group_size kernel attribute in SYCL 2020.

#include "sycl.hpp"

using namespace sycl;
queue q;

class Functor16x16x16 {
public:
  [[sycl::reqd_work_group_size(16, 16, 16)]] void operator()() const {}
};

class FunctorAttr {
public:
  __attribute__((reqd_work_group_size(128, 128, 128))) void operator()() const {} // expected-warning {{attribute 'reqd_work_group_size' is deprecated}} \
                                                                                  // expected-note {{did you mean to use '[[sycl::reqd_work_group_size]]' instead?}}
};

struct TRIFuncObjGood {
  [[sycl::reqd_work_group_size(1, 2, 3)]] void
  operator()() const;
};

[[sycl::reqd_work_group_size(1, 2, 3)]] void TRIFuncObjGood::operator()() const {}

class Functor16 {
public:
  [[sycl::reqd_work_group_size(16)]] void operator()() const {}
};

class Functor64 {
public:
  [[sycl::reqd_work_group_size(64, 64)]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name1
    // CHECK: SYCLReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 16
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 16
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 16
    Functor16x16x16 f16x16x16;
    h.single_task<class kernel_name1>(f16x16x16);

    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name2
    // CHECK: SYCLReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 128
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 128
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 128
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 128
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 128
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 128
    FunctorAttr fattr;
    h.single_task<class kernel_name2>(fattr);

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
    // CHECK:       ReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    h.single_task<class test_kernel3>(TRIFuncObjGood());

    // CHECK: FunctionDecl {{.*}} {{.*}}test_kernel4
    // CHECK: SYCLReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 16
    Functor16 f16;
    h.single_task<class test_kernel4>(f16);

    // CHECK: FunctionDecl {{.*}} {{.*}}test_kernel5
    // CHECK: SYCLReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 64
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 64
    Functor64 f64;
    h.single_task<class test_kernel5>(f64);
  });
  return 0;
}
