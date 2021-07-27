// RUN: %clang_cc1 %s -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -fsyntax-only -sycl-std=2020 -Wno-sycl-2017-compat -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 %s -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -fsyntax-only -sycl-std=2017 -Wno-sycl-2017-compat -ast-dump | FileCheck %s

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
// If the declaration has a [[intel::reqd_work_group_size]]
// [[cl::reqd_work_group_size]] attribute, tests that check
// if the work group size attribute argument (the last argument)
// can be evenly divided by the [[intel::num_simd_work_items()]] attribute.
struct TRIFuncObjBad1 {
  [[intel::num_simd_work_items(3)]]        // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  [[sycl::reqd_work_group_size(3, 6, 5)]] // expected-note{{conflicting attribute is here}}
  void
  operator()() const {}
};

struct TRIFuncObjBad2 {
  [[sycl::reqd_work_group_size(3, 6, 5)]] // expected-note{{conflicting attribute is here}}
  [[intel::num_simd_work_items(3)]]        // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  void
  operator()() const {}
};

// Tests for the default values of [[intel::reqd_work_group_size()]].

// FIXME: This should be accepted instead of error which turns out to be
// an implementation bug that shouldn't be visible to the user as there
// aren't really any default values. The dimensionality of the attribute
// must match the kernel, so three different forms of the attribute
// (one, two, and three argument) can be used instead of assuming default
// values. This will prevent to redeclare the function with a different dimensionality.
struct TRIFuncObjBad3 {
  [[intel::num_simd_work_items(3)]]  // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  [[sycl::reqd_work_group_size(3)]] //expected-note{{conflicting attribute is here}}
  void
  operator()() const {}
};

// FIXME: This should be accepted instead of error which turns out to be
// an implementation bug that shouldn't be visible to the user as there
// aren't really any default values. The dimensionality of the attribute
// must match the kernel, so three different forms of the attribute
// (one, two, and three argument) can be used instead of assuming default
// values. This will prevent to redeclare the function with a different dimensionality.
struct TRIFuncObjBad4 {
  [[sycl::reqd_work_group_size(3)]] // expected-note{{conflicting attribute is here}}
  [[intel::num_simd_work_items(3)]]  // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  void
  operator()() const {}
};

// FIXME: This should be accepted instead of error which turns out to be
// an implementation bug that shouldn't be visible to the user as there
// aren't really any default values. The dimensionality of the attribute
// must match the kernel, so three different forms of the attribute
// (one, two, and three argument) can be used instead of assuming default
// values. This will prevent to redeclare the function with a different dimensionality.
struct TRIFuncObjBad5 {
  [[intel::num_simd_work_items(4)]]      // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  [[sycl::reqd_work_group_size(4, 64)]] // expected-note{{conflicting attribute is here}}
  void
  operator()() const {}
};

// FIXME: This should be accepted instead of error which turns out to be
// an implementation bug that shouldn't be visible to the user as there
// aren't really any default values. The dimensionality of the attribute
// must match the kernel, so three different forms of the attribute
// (one, two, and three argument) can be used instead of assuming default
// values. This will prevent to redeclare the function with a different dimensionality.
struct TRIFuncObjBad6 {
  [[sycl::reqd_work_group_size(4, 64)]] // expected-note{{conflicting attribute is here}}
  [[intel::num_simd_work_items(4)]]      // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  void
  operator()() const {}
};

struct TRIFuncObjBad7 {
  [[sycl::reqd_work_group_size(6, 3, 5)]] // expected-note{{conflicting attribute is here}}
  [[intel::num_simd_work_items(3)]]     // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  void
  operator()() const {}
};

struct TRIFuncObjBad8 {
  [[intel::num_simd_work_items(3)]]     // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  [[sycl::reqd_work_group_size(6, 3, 5)]] // expected-note{{conflicting attribute is here}}
  void
  operator()() const {}
};

[[intel::num_simd_work_items(2)]] // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
[[sycl::reqd_work_group_size(4, 2, 3)]] void func1(); // expected-note{{conflicting attribute is here}}

[[intel::reqd_work_group_size(4, 2, 3)]] // expected-note{{conflicting attribute is here}} \
                                         // expected-warning {{attribute 'intel::reqd_work_group_size' is deprecated}} \
                                         // expected-note {{did you mean to use 'sycl::reqd_work_group_size' instead?}}
[[intel::num_simd_work_items(2)]] void func2(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

[[intel::num_simd_work_items(2)]] // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
[[cl::reqd_work_group_size(4, 2, 3)]] void func3(); // expected-note{{conflicting attribute is here}} expected-warning {{attribute 'cl::reqd_work_group_size' is deprecated}} expected-note {{did you mean to use 'sycl::reqd_work_group_size' instead?}}

[[cl::reqd_work_group_size(4, 2, 3)]] // expected-note{{conflicting attribute is here}} expected-warning {{attribute 'cl::reqd_work_group_size' is deprecated}} expected-note {{did you mean to use 'sycl::reqd_work_group_size' instead?}}
[[intel::num_simd_work_items(2)]] void func4(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

// If the declaration has a __attribute__((reqd_work_group_size()))
// attribute, tests that check if the work group size attribute argument
// (the last argument) can be evenly divided by the [[intel::num_simd_work_items()]]
// attribute.
[[intel::num_simd_work_items(2)]] // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
__attribute__((reqd_work_group_size(4, 2, 5))) void func5(); // expected-note{{conflicting attribute is here}} expected-warning {{attribute 'reqd_work_group_size' is deprecated}} expected-note {{did you mean to use '[[sycl::reqd_work_group_size]]' instead?}}

// Tests for incorrect argument values for Intel FPGA num_simd_work_items and reqd_work_group_size function attributes
struct TRIFuncObjBad9 {
  [[sycl::reqd_work_group_size(5, 5, 5)]]
  [[intel::num_simd_work_items(0)]] // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  void operator()() const {}
};

struct TRIFuncObjBad10 {
  [[intel::num_simd_work_items(0)]] // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  [[sycl::reqd_work_group_size(5, 5, 5)]] void
  operator()() const {}
};

struct TRIFuncObjBad11 {
  [[intel::num_simd_work_items(3.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  [[sycl::reqd_work_group_size(64, 64, 64)]]
  void operator()() const {}
};

struct TRIFuncObjBad12 {
  [[sycl::reqd_work_group_size(64, 64, 64)]]
  [[intel::num_simd_work_items(3.f)]]  // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  void operator()() const {}
};

struct TRIFuncObjBad13 {
  [[sycl::reqd_work_group_size(0)]] // expected-error{{'reqd_work_group_size' attribute must be greater than 0}}
  [[intel::num_simd_work_items(0)]]  // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  void operator()() const {}
};

struct TRIFuncObjBad14 {
  [[intel::num_simd_work_items(0)]]  // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  [[sycl::reqd_work_group_size(0)]] // expected-error{{'reqd_work_group_size' attribute must be greater than 0}}
  void operator()() const {}
};

struct TRIFuncObjBad15 {
  [[intel::num_simd_work_items(3.f)]]  // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  [[sycl::reqd_work_group_size(3.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  void operator()() const {}
};

struct TRIFuncObjBad16 {
  [[sycl::reqd_work_group_size(3.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  [[intel::num_simd_work_items(3.f)]]  // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  void operator()() const {}
};

struct TRIFuncObjBad17 {
  [[intel::num_simd_work_items(3)]]
  [[sycl::reqd_work_group_size(3, 3, 3.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  void operator()() const {}
};

struct TRIFuncObjBad18 {
  [[intel::num_simd_work_items(-1)]]  // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  [[sycl::reqd_work_group_size(-1)]] // expected-warning{{implicit conversion changes signedness: 'int' to 'unsigned long long'}}
  void operator()() const {}
};

#endif // TRIGGER_ERROR
// If the declaration has a [[intel::reqd_work_group_size()]]
// or [[cl::reqd_work_group_size()]] or
// __attribute__((reqd_work_group_size)) attribute, check to see
// if the last argument can be evenly divided by the
// [[intel::num_simd_work_items()]] attribute.
struct TRIFuncObjGood1 {
  [[intel::num_simd_work_items(4)]]
  [[sycl::reqd_work_group_size(3, 64, 4)]] void
  operator()() const {}
};

struct TRIFuncObjGood2 {
  [[sycl::reqd_work_group_size(3, 64, 4)]]
  [[intel::num_simd_work_items(4)]] void
  operator()() const {}
};

struct TRIFuncObjGood3 {
  [[intel::num_simd_work_items(4)]]
  [[sycl::reqd_work_group_size(3, 64, 4)]] void
  operator()() const {}
};

struct TRIFuncObjGood4 {
  [[sycl::reqd_work_group_size(3, 64, 4)]]
  [[intel::num_simd_work_items(4)]] void
  operator()() const {}
};

[[intel::num_simd_work_items(2)]]
__attribute__((reqd_work_group_size(3, 2, 6))) void func6(); // expected-warning {{attribute 'reqd_work_group_size' is deprecated}} \
                                                                expected-note {{did you mean to use '[[sycl::reqd_work_group_size]]' instead?}}

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
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel5>(TRIFuncObjGood2());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel5
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
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
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel7>(TRIFuncObjGood4());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel7
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

#ifdef TRIGGER_ERROR
    [[intel::num_simd_work_items(0)]] int Var = 0; // expected-error{{'num_simd_work_items' attribute only applies to functions}}

    h.single_task<class test_kernel8>(
        []() [[intel::num_simd_work_items(0)]]{}); // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}

    h.single_task<class test_kernel9>(
        []() [[intel::num_simd_work_items(-42)]]{}); // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}

    h.single_task<class test_kernel10>(TRIFuncObjBad1());

    h.single_task<class test_kernel11>(TRIFuncObjBad2());

    h.single_task<class test_kernel12>(TRIFuncObjBad3());

    h.single_task<class test_kernel13>(TRIFuncObjBad4());

    h.single_task<class test_kernel14>(TRIFuncObjBad5());

    h.single_task<class test_kernel15>(TRIFuncObjBad6());

    h.single_task<class test_kernel16>(TRIFuncObjBad7());

    h.single_task<class test_kernel17>(TRIFuncObjBad8());

    h.single_task<class test_kernel18>(TRIFuncObjBad9());

    h.single_task<class test_kernel19>(TRIFuncObjBad10());

    h.single_task<class test_kernel20>(TRIFuncObjBad11());

    h.single_task<class test_kernel21>(TRIFuncObjBad12());

    h.single_task<class test_kernel22>(TRIFuncObjBad13());

    h.single_task<class test_kernel23>(TRIFuncObjBad14());

    h.single_task<class test_kernel24>(TRIFuncObjBad15());

    h.single_task<class test_kernel25>(TRIFuncObjBad16());

    h.single_task<class test_kernel26>(TRIFuncObjBad17());

    h.single_task<class test_kernel27>(TRIFuncObjBad18());

    h.single_task<class test_kernel28>(
        []() [[intel::num_simd_work_items(1), intel::num_simd_work_items(2)]]{}); // expected-warning{{attribute 'num_simd_work_items' is already applied with different arguments}} \
                                                                                  // expected-note {{previous attribute is here}}
#endif // TRIGGER_ERROR
  });
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__
