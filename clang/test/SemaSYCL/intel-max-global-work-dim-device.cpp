// RUN: %clang_cc1 %s -fsyntax-only -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -sycl-std=2017 -triple spir64 -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 %s -fsyntax-only -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -sycl-std=2020 -triple spir64 -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -sycl-std=2017 -triple spir64 -DSYCL2017 %s
// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -sycl-std=2020 -triple spir64 -DSYCL2020 %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

#ifndef __SYCL_DEVICE_ONLY__
struct FuncObj {
  [[intel::max_global_work_dim(1)]] // expected-no-diagnostics
  void
  operator()() const {}
};

void foo() {
  q.submit([&](handler &h) {
    h.single_task<class test_kernel1>(FuncObj());
  });
}

#else // __SYCL_DEVICE_ONLY__

[[intel::max_global_work_dim(2)]] void func_do_not_ignore() {}

struct FuncObj {
  [[intel::max_global_work_dim(1)]] void operator()() const {}
};

struct Func {
  // expected-warning@+2 {{attribute 'intelfpga::max_global_work_dim' is deprecated}}
  // expected-note@+1 {{did you mean to use 'intel::max_global_work_dim' instead?}}
  [[intelfpga::max_global_work_dim(2)]] void operator()() const {}
};

struct TRIFuncObjGood1 {
  [[intel::max_global_work_dim(0)]]
  [[intel::max_work_group_size(1, 1, 1)]]
  [[cl::reqd_work_group_size(1, 1, 1)]] void
  operator()() const {}
};

struct TRIFuncObjGood2 {
  [[intel::max_global_work_dim(3)]]
  [[intel::max_work_group_size(8, 1, 1)]]
  [[cl::reqd_work_group_size(4, 1, 1)]] void
  operator()() const {}
};

#ifdef TRIGGER_ERROR
struct TRIFuncObjBad {
  [[intel::max_global_work_dim(0)]]
  [[intel::max_work_group_size(8, 8, 8)]] // expected-error{{'max_work_group_size' X-, Y- and Z- sizes must be 1 when 'max_global_work_dim' attribute is used with value 0}}
  [[cl::reqd_work_group_size(4, 4, 4)]]   // expected-error{{'reqd_work_group_size' X-, Y- and Z- sizes must be 1 when 'max_global_work_dim' attribute is used with value 0}}
  void
  operator()() const {}
};
#endif // TRIGGER_ERROR

int main() {
  q.submit([&](handler &h) {
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
    // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    h.single_task<class test_kernel1>(FuncObj());

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
    // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // expected-warning@+3 {{attribute 'intelfpga::max_global_work_dim' is deprecated}}
    // expected-note@+2 {{did you mean to use 'intel::max_global_work_dim' instead?}}
    h.single_task<class test_kernel2>(
        []() [[intelfpga::max_global_work_dim(2)]]{});

#if defined(SYCL2020)
    // Test attribute is not propagated.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
    // CHECK-NOT:   SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    h.single_task<class test_kernel3>(
        []() { func_do_not_ignore(); });
#endif // SYCL2020

#if defined(SYCL2017)
    // Test attribute is propagated.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
    // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    h.single_task<class test_kernel3>(
        []() { func_do_not_ignore(); });

    h.single_task<class test_kernel4>(TRIFuncObjGood1());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel4
    // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 0
    // CHECK-NEXT:  IntegerLiteral{{.*}}0{{$}}
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}

    h.single_task<class test_kernel5>(TRIFuncObjGood2());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel5
    // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
#endif // SYCL2017
#ifdef TRIGGER_ERROR
    [[intel::max_global_work_dim(1)]] int Var = 0; // expected-error{{'max_global_work_dim' attribute only applies to functions}}

    h.single_task<class test_kernel6>(
        []() [[intel::max_global_work_dim(-8)]]{}); // expected-error{{'max_global_work_dim' attribute requires a non-negative integral compile time constant expression}}

    h.single_task<class test_kernel7>(
        []() [[intel::max_global_work_dim(3),
               intel::max_global_work_dim(2)]]{}); // expected-warning{{attribute 'max_global_work_dim' is already applied with different arguments}}

    h.single_task<class test_kernel8>(TRIFuncObjBad());

    h.single_task<class test_kernel9>(
        []() [[intel::max_global_work_dim(4)]]{}); // expected-error{{'max_global_work_dim' attribute requires integer constant between 0 and 3 inclusive}}
#endif // TRIGGER_ERROR
  });
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__
