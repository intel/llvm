// RUN: %clang_cc1 %s -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -fsyntax-only -Wno-sycl-2017-compat -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 %s -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -fsyntax-only -Wno-sycl-2017-compat -ast-dump | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

#ifndef __SYCL_DEVICE_ONLY__
struct FuncObj {
  [[intel::num_simd_work_items(42)]]
  void
  operator()() const {}
};

struct FuncObj {
  // expected-warning@+2 {{attribute 'intelfpga::num_simd_work_items' is deprecated}}
  // expected-note@+1 {{did you mean to use 'intel::num_simd_work_items' instead?}}
  [[intelfpga::num_simd_work_items(42)]] void
  operator()() const {}
};

void foo() {
  q.submit([&](handler &h) {
    h.single_task<class test_kernel1>(FuncObj());
  });
}

// No diagnostic is emitted because the arguments match.
[[intel::num_simd_work_items(12)]] void bar();
[[intel::num_simd_work_items(12)]] void bar() {} // OK

// Diagnostic is emitted because the arguments mismatch.
[[intel::num_simd_work_items(12)]] void baz();  // expected-note {{previous attribute is here}}
[[intel::num_simd_work_items(100)]] void baz(); // expected-warning {{attribute 'num_simd_work_items' is already applied with different parameters}}

#else // __SYCL_DEVICE_ONLY__
[[intel::num_simd_work_items(2)]] void func_do_not_ignore() {}

struct FuncObj {
  [[intel::num_simd_work_items(42)]] void operator()() const {}
};

#ifdef TRIGGER_ERROR
// If the declaration has an [[intel::reqd_work_group_size]] or
// [[cl::reqd_work_group_size]] attribute, tests that check if
// the work group size attribute argument (the first argument)
// can be evenly divided by the num_simd_work_items attribute.
struct TRIFuncObjBad1 {
  [[intel::num_simd_work_items(3)]]        // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  [[intel::reqd_work_group_size(5, 3, 3)]] // expected-note{{conflicting attribute is here}}
  void
  operator()() const {}
};

struct TRIFuncObjBad2 {
  [[intel::reqd_work_group_size(5, 3, 3)]] // expected-note{{conflicting attribute is here}}
  [[intel::num_simd_work_items(3)]]        // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  void
  operator()() const {}
};

struct TRIFuncObjBad3 {
  [[intel::num_simd_work_items(3)]]     // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  [[sycl::reqd_work_group_size(5, 3, 3)]] // expected-note{{conflicting attribute is here}}
  void
  operator()() const {}
};

struct TRIFuncObjBad4 {
  [[sycl::reqd_work_group_size(5, 3, 3)]] // expected-note{{conflicting attribute is here}}
  [[intel::num_simd_work_items(3)]]     // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  void
  operator()() const {}
};

struct TRIFuncObjBad5 {
  [[intel::num_simd_work_items(3)]]  // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  [[intel::reqd_work_group_size(5)]] //expected-note{{conflicting attribute is here}}
  void
  operator()() const {}
};

struct TRIFuncObjBad6 {
  [[intel::reqd_work_group_size(5)]] // expected-note{{conflicting attribute is here}}
  [[intel::num_simd_work_items(3)]]  // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  void
  operator()() const {}
};

struct TRIFuncObjBad7 {
  [[intel::num_simd_work_items(4)]]      // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  [[intel::reqd_work_group_size(3, 64)]] // expected-note{{conflicting attribute is here}}
  void
  operator()() const {}
};

struct TRIFuncObjBad8 {
  [[intel::reqd_work_group_size(3, 64)]] // expected-note{{conflicting attribute is here}}
  [[intel::num_simd_work_items(4)]]      // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  void
  operator()() const {}
};

// Tests for incorrect argument values for Intel FPGA num_simd_work_items and reqd_work_group_size function attributes
struct TRIFuncObjBad9 {
  [[intel::reqd_work_group_size(5, 5, 5)]]
  [[intel::num_simd_work_items(0)]] // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  void operator()() const {}
};

struct TRIFuncObjBad10 {
  [[intel::num_simd_work_items(0)]] // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  [[intel::reqd_work_group_size(5, 5, 5)]] void
  operator()() const {}
};

struct TRIFuncObjBad11 {
  [[intel::num_simd_work_items(3.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  [[intel::reqd_work_group_size(64, 64, 64)]]
  void operator()() const {}
};

struct TRIFuncObjBad12 {
  [[intel::reqd_work_group_size(64, 64, 64)]]
  [[intel::num_simd_work_items(3.f)]]  // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  void operator()() const {}
};

struct TRIFuncObjBad13 {
  [[intel::reqd_work_group_size(0)]] // expected-error{{'reqd_work_group_size' attribute must be greater than 0}}
  [[intel::num_simd_work_items(0)]]  // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  void operator()() const {}
};

struct TRIFuncObjBad14 {
  [[intel::num_simd_work_items(0)]]  // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  [[intel::reqd_work_group_size(0)]] // expected-error{{'reqd_work_group_size' attribute must be greater than 0}}
  void operator()() const {}
};

struct TRIFuncObjBad15 {
  [[intel::num_simd_work_items(3.f)]]  // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  [[intel::reqd_work_group_size(3.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  void operator()() const {}
};

struct TRIFuncObjBad16 {
  [[intel::reqd_work_group_size(3.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  [[intel::num_simd_work_items(3.f)]]  // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  void operator()() const {}
};

struct TRIFuncObjBad17 {
  [[intel::num_simd_work_items(3)]]
  [[intel::reqd_work_group_size(3, 3, 3.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  void operator()() const {}
};

struct TRIFuncObjBad18 {
  [[intel::num_simd_work_items(-1)]]  // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  [[intel::reqd_work_group_size(-1)]] // expected-warning{{implicit conversion changes signedness: 'int' to 'unsigned long long'}}
  void operator()() const {}
};

#endif // TRIGGER_ERROR
// If the declaration has an [[intel::reqd_work_group_size]] or
// [[cl::reqd_work_group_size]] attribute, tests that check if
// the work group size attribute argument (the first argument)
// can be evenly divided by the num_simd_work_items attribute.
struct TRIFuncObjGood1 {
  [[intel::num_simd_work_items(4)]]
  [[intel::reqd_work_group_size(64, 64, 5)]] void
  operator()() const {}
};

struct TRIFuncObjGood2 {
  [[intel::reqd_work_group_size(64, 64, 5)]]
  [[intel::num_simd_work_items(4)]] void
  operator()() const {}
};

struct TRIFuncObjGood3 {
  [[intel::num_simd_work_items(4)]]
  [[sycl::reqd_work_group_size(64, 64, 5)]] void
  operator()() const {}
};

struct TRIFuncObjGood4 {
  [[sycl::reqd_work_group_size(64, 64, 5)]]
  [[intel::num_simd_work_items(4)]] void
  operator()() const {}
};

struct TRIFuncObjGood5 {
  [[intel::num_simd_work_items(4)]]
  [[intel::reqd_work_group_size(64)]] void
  operator()() const {}
};

struct TRIFuncObjGood6 {
  [[intel::reqd_work_group_size(64)]]
  [[intel::num_simd_work_items(4)]] void
  operator()() const {}
};

struct TRIFuncObjGood7 {
  [[intel::num_simd_work_items(4)]]
  [[intel::reqd_work_group_size(64, 5)]] void
  operator()() const {}
};

struct TRIFuncObjGood8 {
  [[intel::reqd_work_group_size(64, 5)]]
  [[intel::num_simd_work_items(4)]] void
  operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 42
    // CHECK-NEXT:  IntegerLiteral{{.*}}42{{$}}
    h.single_task<class test_kernel1>(FuncObj());

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // expected-warning@+3 {{attribute 'intelfpga::num_simd_work_items' is deprecated}}
    // expected-note@+2 {{did you mean to use 'intel::num_simd_work_items' instead?}}
    h.single_task<class test_kernel2>(
        []() [[intelfpga::num_simd_work_items(8)]]{});

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    h.single_task<class test_kernel3>(
        []() { func_do_not_ignore(); });

    h.single_task<class test_kernel4>(TRIFuncObjGood1());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel4
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 5
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}

    h.single_task<class test_kernel5>(TRIFuncObjGood2());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel5
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 5
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel6>(TRIFuncObjGood3());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel6
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 5
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}

    h.single_task<class test_kernel7>(TRIFuncObjGood4());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel7
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 5
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel8>(TRIFuncObjGood5());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel8
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}

    h.single_task<class test_kernel9>(TRIFuncObjGood6());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel9
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel10>(TRIFuncObjGood7());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel10
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 5
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}

    h.single_task<class test_kernel11>(TRIFuncObjGood8());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel11
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 5
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

#ifdef TRIGGER_ERROR
    [[intel::num_simd_work_items(0)]] int Var = 0; // expected-error{{'num_simd_work_items' attribute only applies to functions}}

    h.single_task<class test_kernel12>(
        []() [[intel::num_simd_work_items(0)]]{}); // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}

    h.single_task<class test_kernel13>(
        []() [[intel::num_simd_work_items(-42)]]{}); // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}

    h.single_task<class test_kernel14>(TRIFuncObjBad1());

    h.single_task<class test_kernel15>(TRIFuncObjBad2());

    h.single_task<class test_kernel16>(TRIFuncObjBad3());

    h.single_task<class test_kernel17>(TRIFuncObjBad4());

    h.single_task<class test_kernel18>(TRIFuncObjBad5());

    h.single_task<class test_kernel19>(TRIFuncObjBad6());

    h.single_task<class test_kernel20>(TRIFuncObjBad7());

    h.single_task<class test_kernel21>(TRIFuncObjBad8());

    h.single_task<class test_kernel22>(TRIFuncObjBad9());

    h.single_task<class test_kernel23>(TRIFuncObjBad10());

    h.single_task<class test_kernel24>(TRIFuncObjBad11());

    h.single_task<class test_kernel25>(TRIFuncObjBad12());

    h.single_task<class test_kernel26>(TRIFuncObjBad13());

    h.single_task<class test_kernel27>(TRIFuncObjBad14());

    h.single_task<class test_kernel28>(TRIFuncObjBad15());

    h.single_task<class test_kernel29>(TRIFuncObjBad16());

    h.single_task<class test_kernel30>(TRIFuncObjBad17());

    h.single_task<class test_kernel31>(TRIFuncObjBad18());

    h.single_task<class test_kernel32>(
        []() [[intel::num_simd_work_items(1), intel::num_simd_work_items(2)]]{}); // expected-warning{{attribute 'num_simd_work_items' is already applied with different arguments}} \
                                                                                  // expected-note {{previous attribute is here}}
#endif // TRIGGER_ERROR
  });
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__
