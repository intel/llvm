// RUN: %clang_cc1 %s -fsyntax-only -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -triple spir64 -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2017 -triple spir64 | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

#ifndef __SYCL_DEVICE_ONLY__
struct FuncObj {
  [[intel::max_work_group_size(1, 1, 1)]] // expected-no-diagnostics
  void
  operator()() const {}
};

void foo() {
  q.submit([&](handler &h) {
    h.single_task<class test_kernel1>(FuncObj());
  });
}

#else // __SYCL_DEVICE_ONLY__

[[intel::max_work_group_size(2, 2, 2)]] void func_do_not_ignore() {}

struct FuncObj {
  [[intel::max_work_group_size(4, 4, 4)]] void operator()() const {}
};

struct Func {
  // expected-warning@+2 {{attribute 'intelfpga::max_work_group_size' is deprecated}}
  // expected-note@+1 {{did you mean to use 'intel::max_work_group_size' instead?}}
  [[intelfpga::max_work_group_size(1, 1, 1)]] void operator()() const {}
};

#ifdef TRIGGER_ERROR
struct DAFuncObj {
  [[intel::max_work_group_size(4, 4, 4)]]
  [[cl::reqd_work_group_size(8, 8, 4)]] // expected-error{{'reqd_work_group_size' attribute conflicts with 'max_work_group_size' attribute}} \
                                        // expected-warning{{attribute 'cl::reqd_work_group_size' is deprecated}} \
                                        // expected-note{{did you mean to use 'sycl::reqd_work_group_size' instead?}}
  void operator()() const {}
};
#endif // TRIGGER_ERROR

int main() {
  q.submit([&](handler &h) {
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    h.single_task<class test_kernel1>(FuncObj());

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}} {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // expected-warning@+3 {{attribute 'intelfpga::max_work_group_size' is deprecated}}
    // expected-note@+2 {{did you mean to use 'intel::max_work_group_size' instead?}}
    h.single_task<class test_kernel2>(
        []() [[intelfpga::max_work_group_size(8, 8, 8)]]{});

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    h.single_task<class test_kernel3>(
        []() { func_do_not_ignore(); });

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel4
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int -8
    // CHECK-NEXT:  UnaryOperator{{.*}} 'int' prefix '-'
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // expected-warning@+2{{implicit conversion changes signedness: 'int' to 'unsigned long long'}}
    h.single_task<class test_kernel4>(
        []() [[intel::max_work_group_size(8, 8, -8)]]{});

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel5
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int -8
    // CHECK-NEXT:  UnaryOperator{{.*}} 'int' prefix '-'
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int -8
    // CHECK-NEXT:  UnaryOperator{{.*}} 'int' prefix '-'
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // expected-warning@+2 2{{implicit conversion changes signedness: 'int' to 'unsigned long long'}}
    h.single_task<class test_kernel5>(
        []() [[intel::max_work_group_size(-8, 8, -8)]]{});
#ifdef TRIGGER_ERROR
    [[intel::max_work_group_size(1, 1, 1)]] int Var = 0; // expected-error{{'max_work_group_size' attribute only applies to functions}}

    h.single_task<class test_kernel6>(
        []() [[intel::max_work_group_size(0, 1, 3)]]{}); // expected-error{{'max_work_group_size' attribute must be greater than 0}}

    h.single_task<class test_kernel7>(
        []() [[intel::max_work_group_size(1.2f, 1, 3)]]{}); // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}

    h.single_task<class test_kernel8>(
        []() [[intel::max_work_group_size(16, 16, 16),   // expected-note{{conflicting attribute is here}}
               intel::max_work_group_size(2, 2, 2)]]{}); // expected-warning{{attribute 'max_work_group_size' is already applied with different arguments}}

    h.single_task<class test_kernel9>(
        DAFuncObj());

#endif // TRIGGER_ERROR
  });
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__
