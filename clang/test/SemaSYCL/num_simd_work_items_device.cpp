// RUN: %clang_cc1 %s -fsycl -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -fsyntax-only -Wno-sycl-2017-compat -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 %s -fsycl -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -fsyntax-only -Wno-sycl-2017-compat -ast-dump | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

#ifndef __SYCL_DEVICE_ONLY__
struct FuncObj {
  [[intel::num_simd_work_items(42)]] // expected-no-diagnostics
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

#else // __SYCL_DEVICE_ONLY__
[[intel::num_simd_work_items(2)]] void func_do_not_ignore() {}

struct FuncObj {
  [[intel::num_simd_work_items(42)]] void operator()() const {}
};

#ifdef TRIGGER_ERROR
struct TRIFuncObjBad1 {
  [[intel::num_simd_work_items(3)]]        // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  [[intel::reqd_work_group_size(5, 5, 5)]] //expected-note{{conflicting attribute is here}}
  void
  operator()() const {}
};

struct TRIFuncObjBad2 {
  [[intel::reqd_work_group_size(5, 5, 5)]] // expected-note{{conflicting attribute is here}}
  [[intel::num_simd_work_items(3)]]        // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  void
  operator()() const {}
};

struct TRIFuncObjBad3 {
  [[intel::num_simd_work_items(3)]]     // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  [[cl::reqd_work_group_size(5, 5, 5)]] //expected-note{{conflicting attribute is here}}
  void
  operator()() const {}
};

struct TRIFuncObjBad4 {
  [[cl::reqd_work_group_size(5, 5, 5)]] // expected-note{{conflicting attribute is here}}
  [[intel::num_simd_work_items(3)]]     // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  void
  operator()() const {}
};

struct TRIFuncObjBad5 {
  [[intel::num_simd_work_items(0)]] // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  [[intel::reqd_work_group_size(5, 5, 5)]] void
  operator()() const {}
};

struct TRIFuncObjBad6 {
  [[intel::num_simd_work_items(3)]]  // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  [[intel::reqd_work_group_size(5)]] //expected-note{{conflicting attribute is here}}
  void
  operator()() const {}
};

struct TRIFuncObjBad7 {
  [[intel::reqd_work_group_size(5)]] // expected-note{{conflicting attribute is here}}
  [[intel::num_simd_work_items(3)]]  // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  void
  operator()() const {}
};

struct TRIFuncObjBad8 {
  [[intel::num_simd_work_items(3)]]     // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  [[intel::reqd_work_group_size(5, 5)]] //expected-note{{conflicting attribute is here}}
  void
  operator()() const {}
};

struct TRIFuncObjBad9 {
  [[intel::reqd_work_group_size(5, 5)]] // expected-note{{conflicting attribute is here}}
  [[intel::num_simd_work_items(3)]]     // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  void
  operator()() const {}
};
#endif // TRIGGER_ERROR

struct TRIFuncObjGood1 {
  [[intel::num_simd_work_items(4)]]
  [[intel::reqd_work_group_size(64, 64, 64)]] void
  operator()() const {}
};

struct TRIFuncObjGood2 {
  [[intel::reqd_work_group_size(64, 64, 64)]]
  [[intel::num_simd_work_items(4)]] void
  operator()() const {}
};

struct TRIFuncObjGood3 {
  [[intel::num_simd_work_items(4)]]
  [[cl::reqd_work_group_size(64, 64, 64)]] void
  operator()() const {}
};

struct TRIFuncObjGood4 {
  [[cl::reqd_work_group_size(64, 64, 64)]]
  [[intel::num_simd_work_items(4)]] void
  operator()() const {}
};

struct TRIFuncObjGood5 {
  [[intel::num_simd_work_items(3)]]
  [[intel::max_work_group_size(5, 5, 5)]] void
  operator()() const {}
};

struct TRIFuncObjGood6 {
  [[intel::max_work_group_size(5, 5, 5)]]
  [[intel::num_simd_work_items(3)]] void
  operator()() const {}
};

struct TRIFuncObjGood7 {
  [[intel::num_simd_work_items(4)]]
  [[intel::reqd_work_group_size(64)]] void
  operator()() const {}
};

struct TRIFuncObjGood8 {
  [[intel::reqd_work_group_size(64)]]
  [[intel::num_simd_work_items(4)]] void
  operator()() const {}
};

struct TRIFuncObjGood9 {
  [[intel::num_simd_work_items(4)]]
  [[intel::reqd_work_group_size(64, 64)]] void
  operator()() const {}
};

struct TRIFuncObjGood10 {
  [[intel::reqd_work_group_size(64, 64)]]
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
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel5>(TRIFuncObjGood2());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel5
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel6>(TRIFuncObjGood3());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel6
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel7>(TRIFuncObjGood4());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel7
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel8>(TRIFuncObjGood5());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel8
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}

    h.single_task<class test_kernel9>(TRIFuncObjGood6());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel9
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}

    h.single_task<class test_kernel10>(TRIFuncObjGood7());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel10
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel11>(TRIFuncObjGood8());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel11
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel12>(TRIFuncObjGood9());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel12
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel13>(TRIFuncObjGood10());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel13
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

#ifdef TRIGGER_ERROR
    [[intel::num_simd_work_items(0)]] int Var = 0; // expected-error{{'num_simd_work_items' attribute only applies to functions}}

    h.single_task<class test_kernel14>(
        []() [[intel::num_simd_work_items(0)]]{}); // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}

    h.single_task<class test_kernel15>(
        []() [[intel::num_simd_work_items(-42)]]{}); // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}

    h.single_task<class test_kernel16>(TRIFuncObjBad1());

    h.single_task<class test_kernel17>(TRIFuncObjBad2());

    h.single_task<class test_kernel18>(TRIFuncObjBad3());

    h.single_task<class test_kernel19>(TRIFuncObjBad4());

    h.single_task<class test_kernel20>(TRIFuncObjBad5());

    h.single_task<class test_kernel21>(TRIFuncObjBad6());

    h.single_task<class test_kernel22>(TRIFuncObjBad7());

    h.single_task<class test_kernel13>(TRIFuncObjBad8());

    h.single_task<class test_kernel24>(TRIFuncObjBad9());

    h.single_task<class test_kernel25>(
        []() [[intel::num_simd_work_items(1), intel::num_simd_work_items(2)]]{}); // expected-warning{{attribute 'num_simd_work_items' is already applied with different parameters}}
#endif // TRIGGER_ERROR
  });
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__
