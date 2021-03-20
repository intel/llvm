// RUN: %clang_cc1 %s -fsyntax-only -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -triple spir64 -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -triple spir64 | FileCheck %s

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

// No diagnostic is thrown since arguments match.
[[intel::max_global_work_dim(2)]] void bar();
[[intel::max_global_work_dim(2)]] void bar() {}

// Checking of different argument values.
[[intel::max_global_work_dim(2)]] void baz();  // expected-note {{previous attribute is here}}
[[intel::max_global_work_dim(1)]] void baz();  // expected-warning {{attribute 'max_global_work_dim' is already applied with different arguments}}

// Checks correctness of mutual usage of different work_group_size attributes:
// reqd_work_group_size, max_work_group_size and max_global_work_dim.
// In case the value of 'max_global_work_dim' attribute equals to 0 we shall
// ensure that if max_work_group_size and reqd_work_group_size attributes exist,
// they hold equal values (1, 1, 1).

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

struct TRIFuncObjGood3 {
  [[intel::reqd_work_group_size(1)]]
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

struct TRIFuncObjGood4 {
  [[cl::reqd_work_group_size(1, 1, 1)]]
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

struct TRIFuncObjGood5 {
  [[intel::max_work_group_size(1, 1, 1)]]
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

struct TRIFuncObjGood6 {
  [[intel::reqd_work_group_size(4, 1, 1)]]
  [[intel::max_global_work_dim(3)]] void
  operator()() const {}
};

struct TRIFuncObjGood7 {
  [[cl::reqd_work_group_size(4, 1, 1)]]
  [[intel::max_global_work_dim(3)]] void
  operator()() const {}
};

struct TRIFuncObjGood8 {
  [[intel::max_work_group_size(8, 1, 1)]]
  [[intel::max_global_work_dim(3)]] void
  operator()() const {}
};

#ifdef TRIGGER_ERROR
// Checks correctness of mutual usage of different work_group_size attributes:
// reqd_work_group_size, max_work_group_size and max_global_work_dim.
// In case the value of 'max_global_work_dim' attribute equals to 0 we shall
// ensure that if max_work_group_size and reqd_work_group_size attributes exist,
// they hold equal values (1, 1, 1).

struct TRIFuncObjBad {
  [[intel::max_global_work_dim(0)]]
  [[intel::max_work_group_size(8, 8, 8)]] // expected-error{{'max_work_group_size' X-, Y- and Z- sizes must be 1 when 'max_global_work_dim' attribute is used with value 0}}
  [[cl::reqd_work_group_size(4, 4, 4)]]   // expected-error{{'reqd_work_group_size' X-, Y- and Z- sizes must be 1 when 'max_global_work_dim' attribute is used with value 0}}
  void
  operator()() const {}
};

struct TRIFuncObjBad1 {
  [[intel::max_work_group_size(8, 8, 8)]] // expected-error{{'max_work_group_size' X-, Y- and Z- sizes must be 1 when 'max_global_work_dim' attribute is used with value 0}}
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

struct TRIFuncObjBad2 {
  [[cl::reqd_work_group_size(4, 4, 4)]]   // expected-error{{'reqd_work_group_size' X-, Y- and Z- sizes must be 1 when 'max_global_work_dim' attribute is used with value 0}}
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

struct TRIFuncObjBad3 {
  [[intel::reqd_work_group_size(4)]]   // expected-error{{'reqd_work_group_size' X-, Y- and Z- sizes must be 1 when 'max_global_work_dim' attribute is used with value 0}}
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

// Tests for incorrect argument values for Intel FPGA function attributes:
// reqd_work_group_size, max_work_group_size and max_global_work_dim.

struct TRIFuncObjBad4 {
  // expected-error@+2{{'reqd_work_group_size' X-, Y- and Z- sizes must be 1 when 'max_global_work_dim' attribute is used with value 0}}
  // expected-warning@+1{{implicit conversion changes signedness: 'int' to 'unsigned long long'}}
  [[intel::reqd_work_group_size(-4, 1)]]
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

struct TRIFuncObjBad5 {
  [[intel::max_work_group_size(4, 4, 4.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

struct TRIFuncObjBad6 {
  [[cl::reqd_work_group_size(0, 4, 4)]] // expected-error{{'reqd_work_group_size' attribute must be greater than 0}}
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

struct TRIFuncObjBad7 {
  [[intel::reqd_work_group_size(4)]]
  [[intel::max_global_work_dim(-2)]] // expected-error{{'max_global_work_dim' attribute requires a non-negative integral compile time constant expression}}
  void operator()() const {}
};

struct TRIFuncObjBad8 {
  [[intel::max_work_group_size(4, 4, 4)]]
  [[intel::max_global_work_dim(4.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  void operator()() const {}
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

    h.single_task<class test_kernel5>(TRIFuncObjGood3());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel5
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
    // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 0
    // CHECK-NEXT:  IntegerLiteral{{.*}}0{{$}}

    h.single_task<class test_kernel6>(TRIFuncObjGood4());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel6
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
    // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 0
    // CHECK-NEXT:  IntegerLiteral{{.*}}0{{$}}

    h.single_task<class test_kernel7>(TRIFuncObjGood5());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel7
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
    // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 0
    // CHECK-NEXT:  IntegerLiteral{{.*}}0{{$}}

    h.single_task<class test_kernel8>(TRIFuncObjGood6());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel8
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
    // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}

    h.single_task<class test_kernel9>(TRIFuncObjGood7());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel9
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
    // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}

    h.single_task<class test_kernel10>(TRIFuncObjGood8());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel10
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
    // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}

    // Diagnostic is thrown since arguments mismatch.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel11 'void ()'
    // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    h.single_task<class test_kernel11>(
        []() [[intel::max_global_work_dim(3),      // expected-note {{previous attribute is here}}
               intel::max_global_work_dim(2)]]{}); // expected-warning{{attribute 'max_global_work_dim' is already applied with different arguments}}

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel12 'void ()'
    // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // No diagnostic is thrown since arguments match. Duplicate attribute is silently ignored.
    h.single_task<class test_kernel12>(
        []() [[intel::max_global_work_dim(2),
               intel::max_global_work_dim(2)]]{}); // OK

#ifdef TRIGGER_ERROR
    [[intel::max_global_work_dim(1)]] int Var = 0; // expected-error{{'max_global_work_dim' attribute only applies to functions}}

    h.single_task<class test_kernel13>(
        []() [[intel::max_global_work_dim(-8)]]{}); // expected-error{{'max_global_work_dim' attribute requires a non-negative integral compile time constant expression}}

    h.single_task<class test_kernell4>(
        []() [[intel::max_global_work_dim(3),      // expected-note {{previous attribute is here}}
               intel::max_global_work_dim(2)]]{}); // expected-warning{{attribute 'max_global_work_dim' is already applied with different arguments}}

    h.single_task<class test_kernel15>(TRIFuncObjBad());

    h.single_task<class test_kernel16>(TRIFuncObjBad1());

    h.single_task<class test_kernel17>(TRIFuncObjBad2());

    h.single_task<class test_kernel18>(TRIFuncObjBad3());

    h.single_task<class test_kernel19>(TRIFuncObjBad4());

    h.single_task<class test_kernel20>(TRIFuncObjBad5());

    h.single_task<class test_kernel21>(TRIFuncObjBad6());

    h.single_task<class test_kernel22>(TRIFuncObjBad7());

    h.single_task<class test_kernel23>(TRIFuncObjBad8());

    h.single_task<class test_kernel24>(
        []() [[intel::max_global_work_dim(4)]]{}); // expected-error{{'max_global_work_dim' attribute requires integer constant between 0 and 3 inclusive}}
#endif // TRIGGER_ERROR
  });
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__
