// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -fsyntax-only -verify -DTRIGGER_ERROR %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -ast-dump %s | FileCheck %s

// Test for AST of reqd_work_group_size kernel attribute in SYCL 2020.

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

#ifndef __SYCL_DEVICE_ONLY__
// expected-no-diagnostics
class Functor {
public:
  [[sycl::reqd_work_group_size(4)]] void operator()() const {}
};

void bar() {
  q.submit([&](handler &h) {
    Functor f;
    h.single_task<class kernel_name>(f);
  });
}

#else
#ifdef TRIGGER_ERROR
class Functor32 {
public:
  [[cl::reqd_work_group_size(32)]] void operator()() const {} // expected-error {{'reqd_work_group_size' attribute requires exactly 3 arguments}} \
                                                              // expected-warning {{attribute 'cl::reqd_work_group_size' is deprecated}} \
                                                              // expected-note {{did you mean to use 'sycl::reqd_work_group_size' instead?}}
};
#endif // TRIGGER_ERROR

class Functor16 {
public:
  [[sycl::reqd_work_group_size(16)]] void operator()() const {}
};

class Functor64 {
public:
  [[sycl::reqd_work_group_size(64, 64)]] void operator()() const {}
};

class Functor16x16x16 {
public:
  [[sycl::reqd_work_group_size(16, 16, 16)]] void operator()() const {}
};

class FunctorAttr {
public:
  __attribute__((reqd_work_group_size(128, 128, 128))) void operator()() const {} // expected-warning {{attribute 'reqd_work_group_size' is deprecated}} \
                                                                                  // expected-note {{did you mean to use '[[sycl::reqd_work_group_size]]' instead?}}
};

int main() {
  q.submit([&](handler &h) {
    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name1
    // CHECK: ReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 16
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 1
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 1
    Functor16 f16;
    h.single_task<class kernel_name1>(f16);

    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name2
    // CHECK: ReqdWorkGroupSizeAttr
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
    h.single_task<class kernel_name2>(f16x16x16);

    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name3
    // CHECK: ReqdWorkGroupSizeAttr
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
    h.single_task<class kernel_name3>(fattr);

    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name4
    // CHECK: ReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 64
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 64
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 1
    Functor64 f64;
    h.single_task<class kernel_name4>(f64);
  });
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__
