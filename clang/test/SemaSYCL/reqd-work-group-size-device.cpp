// RUN: %clang_cc1 %s -fsyntax-only -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -sycl-std=2017 -triple spir64 -DTRIGGER_ERROR -DSYCL2017 -verify %s
// RUN: %clang_cc1 %s -fsyntax-only -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -sycl-std=2020 -triple spir64 -DTRIGGER_ERROR -DSYCL2020 -verify %s
// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -sycl-std=2017 -triple spir64 -DSYCL2017 %s
// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -sycl-std=2020 -triple spir64 -DSYCL2020 %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

[[cl::reqd_work_group_size(16, 16, 16)]] void func() {}

#if defined(SYCL2017)
[[cl::reqd_work_group_size(4, 1, 1)]] void f4x1x1() {} // expected-note {{conflicting attribute is here}}
// expected-note@-1 {{conflicting attribute is here}}
[[cl::reqd_work_group_size(32, 1, 1)]] void f32x1x1() {} // expected-note {{conflicting attribute is here}}

[[cl::reqd_work_group_size(16, 1, 1)]] void f16x1x1() {}   // expected-note {{conflicting attribute is here}}
[[cl::reqd_work_group_size(16, 16, 1)]] void f16x16x1() {} // expected-note {{conflicting attribute is here}}

[[cl::reqd_work_group_size(32, 32, 1)]] void f32x32x1() {}   // expected-note {{conflicting attribute is here}}
[[cl::reqd_work_group_size(32, 32, 32)]] void f32x32x32() {} // expected-note {{conflicting attribute is here}}
#endif // SYCL2017

#if defined(SYCL2020)
[[cl::reqd_work_group_size(4, 1, 1)]] void f4x1x1() {} // OK 
[[cl::reqd_work_group_size(32, 1, 1)]] void f32x1x1() {} // OK

[[cl::reqd_work_group_size(16, 1, 1)]] void f16x1x1() {}   // OK
[[cl::reqd_work_group_size(16, 16, 1)]] void f16x16x1() {} // OK

[[cl::reqd_work_group_size(32, 32, 1)]] void f32x32x1() {}   // OK
[[cl::reqd_work_group_size(32, 32, 32)]] void f32x32x32() {} // OK
#endif // SYCL2020

class Functor16 {
public:
  [[cl::reqd_work_group_size(16, 1, 1)]] [[cl::reqd_work_group_size(16, 1, 1)]] void operator()() const {}
};

#ifdef TRIGGER_ERROR
class Functor32 {
public:
  // expected-note@+3{{conflicting attribute is here}}
  // expected-warning@+2{{attribute 'reqd_work_group_size' is already applied with different arguments}}
  // expected-error@+1{{'reqd_work_group_size' attribute conflicts with 'reqd_work_group_size' attribute}}
  [[cl::reqd_work_group_size(32, 1, 1)]] [[cl::reqd_work_group_size(1, 1, 32)]] void operator()() const {}
};
#endif
class Functor16x16x16 {
public:
  [[cl::reqd_work_group_size(16, 16, 16)]] void operator()() const {}
};

#if defined(SYCL2017)
class Functor8 { // expected-error {{conflicting attributes applied to a SYCL kernel}}
public:
  [[cl::reqd_work_group_size(1, 1, 8)]] void operator()() const { // expected-note {{conflicting attribute is here}}
    f4x1x1();
  }
};

class Functor {
public:
  void operator()() const {
    f4x1x1();
  }
};
#endif // SYCL2017

class FunctorAttr {
public:
  __attribute__((reqd_work_group_size(128, 128, 128))) void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    Functor16 f16;
    h.single_task<class kernel_name1>(f16);

#if defined(SYCL2017)
    // Test attribute is propagated.
    Functor f;
    h.single_task<class kernel_name2>(f);
#endif // SYCL2017

    Functor16x16x16 f16x16x16;
    h.single_task<class kernel_name3>(f16x16x16);

    FunctorAttr fattr;
    h.single_task<class kernel_name4>(fattr);

    h.single_task<class kernel_name5>([]() [[cl::reqd_work_group_size(32, 32, 32), cl::reqd_work_group_size(32, 32, 32)]] {
      f32x32x32();
    });

#ifdef TRIGGER_ERROR
#if defined(SYCL2017)
    // Test attribute is propagated.
    Functor8 f8;
    h.single_task<class kernel_name6>(f8);
#endif // SYCL2017

#if defined(SYCL2017)
    // Test attribute is propagated.
    h.single_task<class kernel_name7>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
      f4x1x1();
      f32x1x1();
    });

    h.single_task<class kernel_name8>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
      f16x1x1();
      f16x16x1();
    });

    h.single_task<class kernel_name9>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
      f32x32x32();
      f32x32x1();
    });
#endif // SYCL2017

    // expected-error@+1 {{expected variable name or 'this' in lambda capture list}}
    h.single_task<class kernel_name10>([[cl::reqd_work_group_size(32, 32, 32)]][]() {
      f32x32x32();
    });

    Functor32 f32;
    h.single_task<class kernel_name11>(f32);

#if defined(SYCL2020)
    // Test attribute is not propagated.
    h.single_task<class kernel_name12>([]() { // OK
      f4x1x1();
      f32x1x1();
    });

    h.single_task<class kernel_name13>([]() { // OK
      f16x1x1();
      f16x16x1();
    });

    h.single_task<class kernel_name14>([]() { // OK
      f32x32x32();
      f32x32x1();
    });
#endif // SYCL2020
#endif // TRIGGER_ERROR

#if defined(SYCL2020)
    // Test attribute is not propagated.
    // CHECK-LABEL: FunctionDecl {{.*}}class kernel_name15
    // CHECK-NOT:   ReqdWorkGroupSizeAttr {{.*}}
    h.single_task<class kernel_name15>(
        []() { func(); });
#endif // SYCL2020

#if defined(SYCL2017)
    // Test attribute is propagated.
    // CHECK-LABEL: FunctionDecl {{.*}}class kernel_name16
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
    h.single_task<class kernel_name16>(
        []() { func(); });
#endif // SYCL2017
  });
  return 0;
}

// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name1
// CHECK: ReqdWorkGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 16
// CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 1
// CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 1
// CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name2
// CHECK: ReqdWorkGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 4
// CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 1
// CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 1
// CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name3
// CHECK: ReqdWorkGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 16
// CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 16
// CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 16
// CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name4
// CHECK: ReqdWorkGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 128
// CHECK-NEXT:  IntegerLiteral{{.*}}128{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 128
// CHECK-NEXT:  IntegerLiteral{{.*}}128{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 128
// CHECK-NEXT:  IntegerLiteral{{.*}}128{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name5
// CHECK: ReqdWorkGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 32
// CHECK-NEXT:  IntegerLiteral{{.*}}32{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 32
// CHECK-NEXT:  IntegerLiteral{{.*}}32{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 32
// CHECK-NEXT:  IntegerLiteral{{.*}}32{{$}}
